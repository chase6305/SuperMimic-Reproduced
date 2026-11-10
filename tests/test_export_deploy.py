"""Small synthetic actor: package, reload, physical rollout and failure provenance."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from test_deployment import make_motion

from supermimic.artifacts import sha256
from supermimic.deployment.core import PolicyRunner
from supermimic.deployment.export import export_policy
from supermimic.deployment.mujoco import deploy
from supermimic.policy.nets import Policy, PolicyCfg


def test_export_reload_and_physical_rollout(tmp_path):
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
    result = deploy(bundle, motion, tmp_path / "rollout", steps=4)
    assert result["status"] == "complete" and result["hardware_deployed"] is False
    assert result["steps"] == 4 and result["contacts"]["physics_substeps"] == 16
    with np.load(tmp_path / "rollout/trajectory.npz") as data:
        assert data["states"].shape == (5, 71)
        assert data["physical_contacts"].shape == (16, 13)
        assert np.isfinite(data["physical_contacts"]).all()
    with (bundle / "policy.ts").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        PolicyRunner(bundle)


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
