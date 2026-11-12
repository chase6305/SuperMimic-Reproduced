"""Worker snapshots preflight source archives and record the verified copy's identity."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from supermimic import worker


def _install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def _worker_config(tmp_path):
    motions = tmp_path / "motions"
    motions.mkdir()
    source = motions / "clip.npz"
    source.write_bytes(b"motion contents")
    output = tmp_path / "run"
    config = tmp_path / "worker.json"
    config.write_text(
        json.dumps(
            {
                "options": {
                    "command": "play",
                    "motions": str(motions),
                    "output": str(output),
                    "viewer": "none",
                },
                "limits": {},
            }
        )
    )
    return config, source, output


def test_invalid_motion_is_rejected_before_snapshot_copy(tmp_path, monkeypatch):
    config, source, output = _worker_config(tmp_path)
    validate = Mock(side_effect=ValueError("invalid archive"))
    snapshot = Mock(side_effect=AssertionError("invalid archive must not be copied"))
    _install_module(monkeypatch, "supermimic.data.schema", validate_motion=validate)
    _install_module(
        monkeypatch, "supermimic.data.discovery", discover_npz=lambda *a, **kw: [source]
    )
    monkeypatch.setattr(worker, "snapshot_file", snapshot)

    assert worker.run_worker(config) == 1

    validate.assert_called_once_with(source)
    snapshot.assert_not_called()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["error"] == "ValueError: invalid archive"
    assert not list((output / "inputs/motions").iterdir())


@pytest.mark.parametrize("quaternion_order", ["xyzw", None])
def test_incompatible_manifest_order_is_rejected_before_validation_or_copy(
    tmp_path, monkeypatch, quaternion_order
):
    config, source, output = _worker_config(tmp_path)
    (source.parent / "manifest.json").write_text(
        json.dumps(
            {
                "robot": "unitree-g1-29dof",
                "quaternion_order": quaternion_order,
                "files": {source.name: {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
            }
        )
    )
    validate = Mock(side_effect=AssertionError("incompatible manifest must not be validated"))
    snapshot = Mock(side_effect=AssertionError("incompatible manifest must not be copied"))
    constrain = Mock(side_effect=AssertionError("incompatible manifest must not load Torch"))
    _install_module(monkeypatch, "supermimic.data.schema", validate_motion=validate)
    _install_module(
        monkeypatch, "supermimic.data.discovery", discover_npz=lambda *a, **kw: [source]
    )
    monkeypatch.setattr(worker, "snapshot_file", snapshot)
    monkeypatch.setattr(worker, "constrain_torch", constrain)

    assert worker.run_worker(config) == 1

    validate.assert_not_called()
    snapshot.assert_not_called()
    constrain.assert_not_called()
    result = json.loads((output / "result.json").read_text())
    assert result["error"] == (
        "ValueError: Prepared motion manifest has an incompatible quaternion order"
    )
    assert not list((output / "inputs/motions").iterdir())


def test_incompatible_manifest_fails_without_importing_numerical_sdks(tmp_path):
    config, source, output = _worker_config(tmp_path)
    (source.parent / "manifest.json").write_text(
        json.dumps({"robot": "unitree-g1-29dof", "quaternion_order": "xyzw", "files": {}})
    )
    package_source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
blocked = {"numpy", "scipy", "torch", "mujoco", "mujoco_warp", "mjlab", "warp"}

class NoSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "supermimic.data.schema" or fullname.partition(".")[0] in blocked:
            raise AssertionError("Invalid manifest imported a numerical module: " + fullname)

sys.meta_path.insert(0, NoSDKs())
from supermimic.worker import run_worker
assert run_worker(Path(sys.argv[2])) == 1
assert "supermimic.data.schema" not in sys.modules
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_source), str(config)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    result = json.loads((output / "result.json").read_text())
    assert result["error"] == (
        "ValueError: Prepared motion manifest has an incompatible quaternion order"
    )


@pytest.mark.parametrize("manifest_order", ["no_manifest", "missing", "wxyz"])
def test_worker_records_snapshot_metadata_and_original_motion_source(
    tmp_path, monkeypatch, manifest_order
):
    config, source, output = _worker_config(tmp_path)
    if manifest_order != "no_manifest":
        manifest = {
            "robot": "unitree-g1-29dof",
            "files": {source.name: {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
        }
        if manifest_order != "missing":
            manifest["quaternion_order"] = manifest_order
        (source.parent / "manifest.json").write_text(json.dumps(manifest))
    snapshot = output / "inputs/motions/clip.npz"
    validated = []

    def validate(path):
        validated.append(path)
        return {"frames": 10 if path == source else 7}

    played = Mock()
    _install_module(monkeypatch, "supermimic.data.schema", validate_motion=validate)
    _install_module(
        monkeypatch, "supermimic.data.discovery", discover_npz=lambda *a, **kw: [source]
    )
    _install_module(monkeypatch, "supermimic.deployment.play", play_checkpoint=played)
    monkeypatch.setattr(worker, "constrain_torch", lambda limits: None)

    assert worker.run_worker(config) == 0

    assert validated == [source, snapshot]
    played.assert_called_once_with(None, [snapshot], 1, "none")
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["motion_files"][source.name] == {
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frames": 7,
        "source": str(source.resolve()),
    }
    assert snapshot.stat().st_mode & 0o222 == 0
    private_manifest = json.loads((snapshot.parent / "manifest.json").read_text())
    assert private_manifest["quaternion_order"] == "wxyz"
