"""Materialized actor weights cannot silently resume an unrelated Adam chain."""

from dataclasses import asdict

import pytest
import torch
from test_transfer import fresh_trainer

from supermimic.checkpoints import validate_optimizer_resume
from supermimic.config import TrainConfig
from supermimic.profiles import training_config
from supermimic.training.loop import Trainer


@pytest.mark.parametrize(
    "block,field",
    [
        ("native_policy_improvement", "policy_optimizer_resume_valid"),
        ("native_policy_improvement", "critic_optimizer_resume_valid"),
        ("native_dynamics_calibration", "optimizer_resume_valid"),
    ],
)
def test_declared_stale_state_is_rejected_before_configuration_or_weights_change(
    tmp_path, block, field
):
    path = tmp_path / "stale.pt"
    # No state dictionaries: validation must reject before any attempted restore.
    torch.save({"config": asdict(TrainConfig()), "iteration": 0, block: {field: False}}, path)
    target = fresh_trainer()
    before = {name: value.clone() for name, value in target.policy.state_dict().items()}
    with pytest.raises(ValueError, match="stale or unverified.*optimizer.*--transfer"):
        Trainer.load(target, path)
    with pytest.raises(ValueError, match="stale or unverified.*optimizer.*--transfer"):
        training_config("smoke", tmp_path, tmp_path / "run", resume=path)
    assert target.iteration == target.num_updates == 0
    assert not target.policy_trainer.opt.state
    for name, value in target.policy.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_parent_provenance_does_not_override_a_valid_new_optimizer_chain():
    validate_optimizer_resume({})
    validate_optimizer_resume(
        {
            "native_policy_improvement": {
                "policy_optimizer_resume_valid": True,
                "critic_optimizer_resume_valid": True,
                "parent_provenance": {"policy_optimizer_resume_valid": False},
            },
            "native_dynamics_calibration": {
                "optimizer_resume_valid": True,
                "optimizer_correspondence_not_proved": False,
            },
        }
    )
    with pytest.raises(ValueError, match="world model optimizer"):
        validate_optimizer_resume(
            {"native_dynamics_calibration": {"optimizer_correspondence_not_proved": True}}
        )


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_invalid_validity_flag_is_not_treated_as_permission_to_resume(value):
    with pytest.raises(ValueError, match="must be boolean"):
        validate_optimizer_resume(
            {"native_policy_improvement": {"policy_optimizer_resume_valid": value}}
        )


def test_explicit_weight_transfer_accepts_stale_history_and_keeps_fresh_state(tmp_path):
    source, target = fresh_trainer(), fresh_trainer()
    path = tmp_path / "source.pt"
    torch.save(
        {
            "config": asdict(TrainConfig()),
            "iteration": 20,
            "num_updates": 12,
            "wm": source.wm.state_dict(),
            "policy": source.policy.state_dict(),
            "critic": source.critic.state_dict(),
            "control_contract": source.control_contract,
            "native_policy_improvement": {"policy_optimizer_resume_valid": False},
        },
        path,
    )
    Trainer.load(target, path, transfer=True)
    cfg = training_config(
        "smoke", tmp_path, tmp_path / "transfer", resume=path, transfer=True, iterations=100
    )
    assert cfg.transfer
    assert target.iteration == target.num_updates == 0
    assert not target.wm_trainer.opt.state
    assert not target.policy_trainer.opt.state and not target.policy_trainer.critic_opt.state
    for name, value in target.policy.state_dict().items():
        torch.testing.assert_close(value, source.policy.state_dict()[name], atol=0, rtol=0)
