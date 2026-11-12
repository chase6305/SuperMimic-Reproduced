"""Environment observation configuration and deployment share the pure G1 interface."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from supermimic import interfaces


@pytest.mark.parametrize("name", ["TRACKED_BODIES", "ANCHOR_BODY", "ROOT_BODY", "FOOT_GEOM_REGEX"])
def test_tracking_environment_keeps_legacy_robot_constant_identity(name):
    from supermimic.envs import tracking_env_cfg

    assert getattr(tracking_env_cfg, name) is getattr(interfaces, name)


@pytest.mark.parametrize("history", [3, 10])
def test_environment_observation_configuration_uses_shared_order_and_requested_history(history):
    from mjlab.envs import mdp

    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg

    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], history_length=history))
    policy = cfg.observations["policy"]

    assert tuple(policy.terms) == tuple(name for name, _ in interfaces.OBS_TERMS)
    assert policy.history_length == history
    assert policy.concatenate_terms and policy.flatten_history_dim and policy.enable_corruption
    assert policy.terms["joint_pos"].func is mdp.joint_pos_rel
    assert policy.terms["joint_pos"].params == {"biased": True}
    assert policy.terms["joint_vel"].func is mdp.joint_vel_rel
    assert policy.terms["base_ang_vel"].func is mdp.base_ang_vel
    assert policy.terms["projected_gravity"].func is mdp.projected_gravity
    assert policy.terms["actions"].func is mdp.last_action
    assert policy.terms["actions"].noise is None
    for name, limits in {
        "joint_pos": (-0.01, 0.01),
        "joint_vel": (-0.5, 0.5),
        "base_ang_vel": (-0.2, 0.2),
        "projected_gravity": (-0.05, 0.05),
    }.items():
        noise = policy.terms[name].noise
        assert (noise.n_min, noise.n_max) == limits


def test_environment_orders_existing_observation_terms_by_shared_interface(monkeypatch):
    from supermimic.envs import tracking_env_cfg

    reordered = tuple(reversed(interfaces.OBS_TERMS))
    monkeypatch.setattr(tracking_env_cfg, "OBS_TERMS", reordered)
    cfg = tracking_env_cfg.make_env_cfg(tracking_env_cfg.SuperMimicEnvCfgArgs(npz_files=[]))

    assert tuple(cfg.observations["policy"].terms) == tuple(name for name, _ in reordered)


@pytest.mark.parametrize("mismatch", ["missing", "extra"])
def test_environment_rejects_observation_term_mismatch_before_reordering(monkeypatch, mismatch):
    from supermimic.envs import tracking_env_cfg

    terms = interfaces.OBS_TERMS
    terms = terms[:-1] if mismatch == "missing" else (*terms, ("unexpected", 1))
    monkeypatch.setattr(tracking_env_cfg, "OBS_TERMS", terms)

    with pytest.raises(ValueError, match="observation terms differ"):
        tracking_env_cfg.make_env_cfg(tracking_env_cfg.SuperMimicEnvCfgArgs(npz_files=[]))


def test_deployment_metrics_imports_without_environment_configuration():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])

class BlockEnvironmentConfiguration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "supermimic.envs.tracking_env_cfg":
            raise AssertionError("Deployment metrics imported environment configuration")

sys.meta_path.insert(0, BlockEnvironmentConfiguration())
from supermimic.deployment.metrics import tracking_metrics
assert callable(tracking_metrics)
assert "supermimic.envs.tracking_env_cfg" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
