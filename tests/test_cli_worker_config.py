"""Each CLI invocation owns its staging config even when two target the same run."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pytest

from supermimic import cli, runtime, supervisor
from supermimic.artifacts import write_json


def test_simultaneous_cli_calls_keep_their_own_worker_payloads(tmp_path, monkeypatch):
    output = tmp_path / "same-run"
    before_write = threading.Barrier(2, timeout=10)
    after_write = threading.Barrier(2, timeout=10)
    after_read = threading.Barrier(2, timeout=10)
    caller = threading.local()
    observed = []

    def staged_write(path, payload):
        # Both callers passed the CLI's existing-output checks before either
        # publication. A fixed shared path would now deterministically mix seeds.
        before_write.wait()
        write_json(path, payload)
        after_write.wait()

    def read_worker_payload(command, actual_output, limits, **kwargs):
        assert actual_output == output and command[-2] == "_worker"
        config_path = Path(command[-1])
        seed = json.loads(config_path.read_text())["options"]["seed"]
        observed.append((caller.seed, config_path, seed))
        # Keep every caller's context alive until all workers consumed input.
        after_read.wait()
        return 0

    monkeypatch.setattr(cli, "write_json", staged_write, raising=False)
    monkeypatch.setattr(runtime, "write_json", staged_write)
    monkeypatch.setattr(supervisor, "supervise", read_worker_payload)

    def invoke(seed):
        caller.seed = seed
        return cli.main(["train", "--output", str(output), "--seed", str(seed)])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke, seed) for seed in (11, 22)]
        assert [future.result(timeout=30) for future in futures] == [0, 0]

    assert len(observed) == 2
    assert {expected for expected, _, _ in observed} == {11, 22}
    assert all(actual == expected for expected, _, actual in observed)
    paths = [path for _, path, _ in observed]
    assert len(set(paths)) == 2
    assert all(path.parent == output.parent and not path.exists() for path in paths)
    assert not output.exists()  # Only the supervisor may claim the run directory.


def test_cli_preserves_a_legacy_staging_file_owned_by_another_invocation(tmp_path, monkeypatch):
    output = tmp_path / "run"
    legacy = tmp_path / ".run.worker.json"
    legacy.write_bytes(b"another invocation's config")
    staged = []

    def supervise(command, actual_output, limits, **kwargs):
        path = Path(command[-1])
        staged.append(path)
        assert path != legacy
        assert json.loads(path.read_text())["options"]["seed"] == 11
        return 0

    monkeypatch.setattr(supervisor, "supervise", supervise)

    assert cli.main(["train", "--output", str(output), "--seed", "11"]) == 0

    assert len(staged) == 1 and not staged[0].exists()
    assert legacy.read_bytes() == b"another invocation's config"
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["success", "write_failure", "consumer_failure"])
def test_temporary_config_cleans_only_its_own_file_on_every_exit(
    tmp_path, monkeypatch, outcome
):
    output = tmp_path / "existing-run"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_bytes(b"existing run data")
    legacy = tmp_path / ".existing-run.worker.json"
    legacy.write_bytes(b"another invocation's config")
    payload = {"options": {"seed": 11}}
    staged = []

    def record_write(path, value):
        staged.append(Path(path))
        write_json(path, value)
        if outcome == "write_failure":
            raise OSError("config write failed")

    monkeypatch.setattr(runtime, "write_json", record_write)
    if outcome == "write_failure":
        expected = pytest.raises(OSError, match="config write failed")
    elif outcome == "consumer_failure":
        expected = pytest.raises(RuntimeError, match="consumer failed")
    else:
        expected = nullcontext()

    with expected:
        with runtime.temporary_worker_config(output, payload) as config:
            assert config == staged[0] and config.parent == output.parent
            assert config != legacy and json.loads(config.read_text()) == payload
            if outcome == "consumer_failure":
                raise RuntimeError("consumer failed")

    assert len(staged) == 1 and not staged[0].exists()
    assert list(output.iterdir()) == [existing]
    assert existing.read_bytes() == b"existing run data"
    assert legacy.read_bytes() == b"another invocation's config"
