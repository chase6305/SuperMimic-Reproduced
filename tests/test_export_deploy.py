"""Small synthetic actor: package, reload, physical rollout and failure provenance."""

import importlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_deployment import make_motion

import supermimic
import supermimic.deployment.mujoco as runtime
from supermimic.artifacts import sha256
from supermimic.deployment.core import PolicyRunner
from supermimic.deployment.export import export_policy
from supermimic.deployment.mujoco import deploy
from supermimic.policy.nets import Policy, PolicyCfg


def test_export_reload_and_physical_rollout(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    pcfg = PolicyCfg(encoder_hidden=(16, 8), decoder_hidden=(16, 8))
    actor = Policy(pcfg).eval()
    checkpoint = tmp_path / "test.pt"
    torch.save(
        {
            "iteration": 0,
            "policy": actor.state_dict(),
            "config": {"policy": asdict(pcfg), "env": {"future_frames": 10, "anchor_obs": False}},
        },
        checkpoint,
    )
    bundle = tmp_path / "export"
    contract = export_policy(checkpoint, bundle)
    assert contract["export_parity_max_abs_error"] < 1e-6
    assert (bundle / "Unitree-BSD-3-Clause.txt").is_file()
    assert contract["hardware_validated"] is False
    runner = PolicyRunner(bundle)
    q = np.array(contract["default_joint_pos"], dtype=np.float32)
    goal = np.zeros(680, dtype=np.float32)
    action, target = runner.act(q, np.zeros(29), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), goal)
    assert action.shape == target.shape == (29,) and np.isfinite(target).all()
    motion = tmp_path / "test-motion.npz"
    make_motion(motion)
    original_motion_hash = sha256(motion)
    replacement = tmp_path / "replacement.npz"
    with np.load(motion, allow_pickle=False) as data:
        arrays = {}
        for name in data.files:
            value = data[name]
            arrays[name] = value[:7].copy() if value.ndim and value.shape[0] == 20 else value.copy()
    np.savez(replacement, **arrays)
    original_snapshot = runtime.snapshot_file
    original_reference = runtime.MotionReference
    scoring = importlib.import_module("supermimic.deployment.tracking_metrics")
    original_tracking = scoring.tracking_metrics
    snapshots, references, tracked = [], [], []

    def copy_then_publish(source, destination, **kwargs):
        digest = original_snapshot(source, destination, **kwargs)
        assert Path(source) == motion
        snapshots.append(Path(destination))
        replacement.replace(motion)
        return digest

    def read_pinned_reference(path, default_q):
        references.append(Path(path))
        assert Path(path) == snapshots[0]
        assert sha256(path) == original_motion_hash
        reference = original_reference(path, default_q)
        assert reference.frames == 20
        return reference

    def score_pinned_motion(states, actions, frames, motion, contract):
        tracked.append(Path(motion))
        assert Path(motion) == snapshots[0]
        assert sha256(motion) == original_motion_hash
        with np.load(motion, allow_pickle=False) as data:
            assert data["joint_pos"].shape == (20, 29)
        return original_tracking(states, actions, frames, motion, contract)

    monkeypatch.setattr(runtime, "snapshot_file", copy_then_publish)
    monkeypatch.setattr(runtime, "MotionReference", read_pinned_reference)
    monkeypatch.setattr(scoring, "tracking_metrics", score_pinned_motion)

    result = deploy(bundle, motion, tmp_path / "rollout", steps=4)
    assert result["status"] == "complete" and result["hardware_deployed"] is False
    assert result["steps"] == 4 and result["contacts"]["physics_substeps"] == 16
    assert result["motion_sha256"] == original_motion_hash != sha256(motion)
    with np.load(motion, allow_pickle=False) as data:
        assert data["joint_pos"].shape == (7, 29)
    assert len(snapshots) == 1 and references == tracked == snapshots
    assert not snapshots[0].exists() and not snapshots[0].parent.exists()
    assert set(result["implementation_sha256"]) == {"mujoco.py", "core.py", "metrics.py"}
    package = Path(supermimic.__file__).parent
    package_sources = result["implementation_package_sha256"]
    for name in (
        "deployment/reference.py",
        "deployment/tracking_metrics.py",
        "deployment/validation.py",
        "interfaces.py",
    ):
        assert package_sources[name] == sha256(package / name)
    with np.load(tmp_path / "rollout/trajectory.npz") as data:
        assert data["states"].shape == (5, 71)
        assert data["physical_contacts"].shape == (16, 13)
        assert np.isfinite(data["physical_contacts"]).all()
    with (bundle / "policy.ts").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        PolicyRunner(bundle)


def test_public_deploy_removes_private_motion_when_consumer_fails(tmp_path, monkeypatch):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"selected motion bytes")
    output = tmp_path / "not-created/rollout"
    model = object()
    runner = SimpleNamespace(contract={"packages": {"mujoco": "test-version"}})
    fake_mujoco = SimpleNamespace(
        __version__="test-version",
        MjModel=SimpleNamespace(from_binary_path=lambda path: model),
    )
    monkeypatch.setattr(runtime, "mujoco", fake_mujoco)
    monkeypatch.setattr(runtime, "PolicyRunner", lambda bundle: runner)
    monkeypatch.setattr(runtime, "validate_model_contract", lambda model, contract: None)
    monkeypatch.setattr(runtime, "validate_motion", lambda path: {"frames": 20})
    snapshots = []

    def fail_after_snapshot(
        actual_runner, actual_model, actual_bundle, pinned_motion, actual_output, steps, viewer
    ):
        assert actual_runner is runner and actual_model is model
        assert actual_bundle == tmp_path / "bundle"
        assert actual_output == output and steps == 4 and viewer is False
        snapshots.append(Path(pinned_motion))
        assert Path(pinned_motion) != motion
        assert Path(pinned_motion).read_bytes() == motion.read_bytes()
        raise RuntimeError("consumer failed")

    monkeypatch.setattr(runtime, "_deploy", fail_after_snapshot)
    with pytest.raises(RuntimeError, match="consumer failed"):
        runtime.deploy(tmp_path / "bundle", motion, output, steps=4)

    assert len(snapshots) == 1
    assert not snapshots[0].exists() and not snapshots[0].parent.exists()
    assert not output.parent.exists()


def test_atomic_checkpoint_publication_cannot_change_export_provenance(tmp_path, monkeypatch):
    pcfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,))
    actor = Policy(pcfg)
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            "iteration": 12,
            "policy": actor.state_dict(),
            "config": {"policy": asdict(pcfg), "env": {"future_frames": 10}},
        },
        checkpoint,
    )
    expected = sha256(checkpoint)
    original_load = torch.load

    def load_and_publish(stream, **kwargs):
        loaded = original_load(stream, **kwargs)
        replacement = tmp_path / "replacement.pt"
        replacement.write_bytes(b"new checkpoint published concurrently")
        replacement.replace(Path(stream.name))
        return loaded

    monkeypatch.setattr(torch, "load", load_and_publish)
    contract = export_policy(checkpoint, tmp_path / "export")
    assert contract["checkpoint_sha256"] == expected
    assert contract["checkpoint_sha256"] != sha256(checkpoint)
    assert contract["checkpoint_iteration"] == 12
