"""Critic ablation is explicit, inherited, recorded and disabled end-to-end."""

from dataclasses import asdict

import pytest
import torch

from supermimic.cli import parser
from supermimic.config import TrainConfig
from supermimic.deployment.export import export_policy
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.profiles import training_config


def test_critic_cli_distinguishes_inherit_enable_and_disable():
    assert parser().parse_args(["train"]).critic is None
    assert parser().parse_args(["train", "--critic"]).critic is True
    assert parser().parse_args(["train", "--no-critic"]).critic is False


def test_critic_legacy_resume_inheritance_and_explicit_overrides(tmp_path):
    saved = asdict(TrainConfig())
    saved["policy_trainer"].pop("use_critic")
    path = tmp_path / "old.pt"
    torch.save({"iteration": 1, "config": saved}, path)
    assert (
        training_config("smoke", tmp_path, tmp_path / "a", resume=path).policy_trainer.use_critic
        is True
    )
    saved["policy_trainer"]["use_critic"] = False
    path = tmp_path / "no-critic.pt"
    torch.save({"iteration": 1, "config": saved}, path)
    assert (
        training_config("smoke", tmp_path, tmp_path / "b", resume=path).policy_trainer.use_critic
        is False
    )
    assert (
        training_config(
            "smoke", tmp_path, tmp_path / "c", resume=path, use_critic=True
        ).policy_trainer.use_critic
        is True
    )
    assert (
        training_config(
            "smoke", tmp_path, tmp_path / "d", use_critic=False
        ).policy_trainer.use_critic
        is False
    )


@pytest.mark.parametrize("invalid", [0, 1, "false"])
def test_critic_option_rejects_non_boolean_programmatic_values(tmp_path, invalid):
    with pytest.raises(ValueError, match="must be boolean"):
        training_config("smoke", tmp_path, tmp_path / "out", use_critic=invalid)


@pytest.mark.parametrize("saved_choice", [False, None])
def test_export_records_disabled_and_legacy_enabled_critic_choice(tmp_path, saved_choice):
    pcfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,))
    actor = Policy(pcfg)
    training = {} if saved_choice is None else {"use_critic": saved_choice}
    checkpoint = tmp_path / "actor.pt"
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
    assert contract["training_metadata"]["imagination_use_critic"] is (
        True if saved_choice is None else False
    )


def test_cli_disabled_critic_is_never_called_during_actor_update(tmp_path, monkeypatch):
    import test_policy_accumulation as fixture

    import supermimic.imagination.policy_trainer as module

    choice = parser().parse_args(["train", "--no-critic"]).critic
    cfg = training_config("smoke", tmp_path, tmp_path / "out", use_critic=choice).policy_trainer
    model = fixture.trainer(1, use_critic=True)
    model.cfg = cfg
    before = model.policy.weight.detach().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled critic was evaluated")

    monkeypatch.setattr(model.critic, "forward", forbidden)
    monkeypatch.setattr(module, "imagine", fixture.synthetic_rollout)
    metrics = model.update(fixture.Buffer())
    assert not torch.equal(model.policy.weight, before)
    assert "critic_loss" not in metrics and not model.critic_opt.state
    assert all(p.grad is None for p in model.critic.parameters())
