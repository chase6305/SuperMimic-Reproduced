"""Runtime imagination horizon is explicit and survives a verified continuation."""

from dataclasses import asdict

import pytest
import torch

from supermimic.cli import parser
from supermimic.config import TrainConfig
from supermimic.deployment.export import export_policy
from supermimic.memory import estimate_training_memory
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.profiles import training_config


def test_cli_horizon_distinguishes_inherit_from_override():
    assert parser().parse_args(["train"]).policy_horizon is None
    assert parser().parse_args(["train", "--policy-horizon", "8"]).policy_horizon == 8


def test_new_run_defaults_and_explicit_horizon(tmp_path):
    assert training_config("smoke", tmp_path, tmp_path).policy_trainer.horizon == 8
    assert training_config("paper-width", tmp_path, tmp_path).policy_trainer.horizon == 32
    assert (
        training_config("paper-width", tmp_path, tmp_path, policy_horizon=8).policy_trainer.horizon
        == 8
    )


def test_resume_inherits_horizon_and_explicit_override_wins(tmp_path):
    saved = asdict(TrainConfig())
    saved["policy_trainer"]["horizon"] = 16
    path = tmp_path / "saved.pt"
    torch.save({"iteration": 1, "config": saved}, path)
    cfg = training_config("paper-width", tmp_path, tmp_path, resume=path)
    assert cfg.policy_trainer.horizon == 16
    cfg = training_config("paper-width", tmp_path, tmp_path, resume=path, policy_horizon=8)
    assert cfg.policy_trainer.horizon == 8
    assert asdict(cfg)["policy_trainer"]["horizon"] == 8


def test_legacy_without_saved_horizon_uses_selected_profile(tmp_path):
    saved = asdict(TrainConfig())
    saved["policy_trainer"].pop("horizon")
    path = tmp_path / "legacy.pt"
    torch.save({"iteration": 1, "config": saved}, path)
    assert training_config("smoke", tmp_path, tmp_path, resume=path).policy_trainer.horizon == 8
    assert (
        training_config("paper-width", tmp_path, tmp_path, resume=path).policy_trainer.horizon == 32
    )


def test_transfer_uses_new_profile_horizon_or_explicit_override(tmp_path):
    saved = asdict(TrainConfig())
    saved["policy_trainer"]["horizon"] = 8
    path = tmp_path / "saved.pt"
    torch.save({"iteration": 1, "config": saved}, path)
    assert (
        training_config(
            "paper-width", tmp_path, tmp_path, resume=path, transfer=True
        ).policy_trainer.horizon
        == 32
    )
    assert (
        training_config(
            "paper-width", tmp_path, tmp_path, resume=path, transfer=True, policy_horizon=16
        ).policy_trainer.horizon
        == 16
    )


@pytest.mark.parametrize("invalid", [0, -1, 65, 1.5, True, "8"])
def test_invalid_horizons_fail_before_training(tmp_path, invalid):
    with pytest.raises(ValueError, match="Policy horizon"):
        training_config("paper-width", tmp_path, tmp_path, policy_horizon=invalid)


@pytest.mark.parametrize("saved_horizon", [8, None])
def test_export_records_actual_horizon_or_unknown_legacy(tmp_path, saved_horizon):
    pcfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,))
    actor = Policy(pcfg)
    checkpoint = tmp_path / "actor.pt"
    training = {} if saved_horizon is None else {"horizon": saved_horizon}
    torch.save(
        {
            "iteration": 1,
            "policy": actor.state_dict(),
            "config": {
                "policy": asdict(pcfg),
                "env": {"future_frames": 10},
                "policy_trainer": training,
            },
        },
        checkpoint,
    )
    contract = export_policy(checkpoint, tmp_path / "export")
    assert contract["training_metadata"]["imagination_horizon"] == saved_horizon


def test_shorter_horizon_reduces_imagination_memory_estimate(tmp_path):
    a = training_config("smoke", tmp_path, tmp_path, policy_horizon=32)
    b = training_config("smoke", tmp_path, tmp_path, policy_horizon=8)
    old, new = estimate_training_memory(a, 7000), estimate_training_memory(b, 7000)
    assert new["estimated_cuda_mib"] < old["estimated_cuda_mib"]
    assert (
        new["components_mib"]["activation_workspace"] * 4
        == old["components_mib"]["activation_workspace"]
    )
    assert new["parameters"] == old["parameters"]
