"""A worker owns the Trainer from construction through its final run record."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from supermimic import worker
from supermimic.config import TrainConfig


def _install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("failure", ["record", "run"])
def test_worker_closes_trainer_if_running_record_or_training_fails(
    tmp_path, monkeypatch, failure
):
    motions = tmp_path / "motions"
    motions.mkdir()
    source = motions / "clip.npz"
    source.write_bytes(b"selected motion")
    output = tmp_path / "run"
    config = tmp_path / "worker.json"
    config.write_text(
        json.dumps(
            {
                "options": {
                    "command": "train",
                    "profile": "smoke",
                    "motions": str(motions),
                    "output": str(output),
                    "iterations": 10,
                    "hours": 1,
                    "seed": 0,
                },
                "limits": {"max_gpu_mib": 2304},
            }
        )
    )
    trainer = SimpleNamespace(
        iteration=0,
        num_updates=0,
        transfer_provenance=None,
        log_dir=output / "training",
        run=Mock(side_effect=RuntimeError("training failed")),
        close=Mock(),
    )
    constructor = Mock(return_value=trainer)
    _install_module(
        monkeypatch, "supermimic.data.discovery", discover_npz=lambda *a, **kw: [source]
    )
    _install_module(
        monkeypatch, "supermimic.data.schema", validate_motion=lambda path: {"frames": 7}
    )
    _install_module(
        monkeypatch, "supermimic.memory", validate_training_budget=lambda *a: {"accepted": True}
    )
    _install_module(
        monkeypatch, "supermimic.profiles", training_config=lambda *a, **kw: TrainConfig()
    )
    _install_module(monkeypatch, "supermimic.training.loop", Trainer=constructor)
    monkeypatch.setattr(worker, "constrain_torch", lambda limits: None)
    original_write_json = worker.write_json

    def write_record(path, value):
        if (
            failure == "record"
            and path == output / "result.json"
            and value.get("status") == "running"
        ):
            raise OSError("running record failed")
        original_write_json(path, value)

    monkeypatch.setattr(worker, "write_json", write_record)

    assert worker.run_worker(config) == 1

    constructor.assert_called_once()
    trainer.close.assert_called_once_with()
    if failure == "record":
        trainer.run.assert_not_called()
    else:
        trainer.run.assert_called_once_with()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["error"] == (
        "OSError: running record failed" if failure == "record" else "RuntimeError: training failed"
    )
