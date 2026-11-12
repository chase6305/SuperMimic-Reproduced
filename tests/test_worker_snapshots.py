"""Worker snapshots preflight source archives and record the verified copy's identity."""

import hashlib
import json
import sys
from types import ModuleType
from unittest.mock import Mock

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


def test_worker_records_snapshot_metadata_and_original_motion_source(tmp_path, monkeypatch):
    config, source, output = _worker_config(tmp_path)
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
