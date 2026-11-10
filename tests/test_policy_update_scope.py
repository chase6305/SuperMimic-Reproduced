"""Selective actor updates retain feedback gradients and resumable Adam state."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from supermimic.cli import parser
from supermimic.config import TrainConfig
from supermimic.deployment.export import export_policy
from supermimic.imagination.policy_trainer import configure_policy_update_scope
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.profiles import training_config


def actor():
    torch.manual_seed(73)
    return Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16, 12)))


def inputs():
    torch.manual_seed(13)
    return torch.randn(4, 930), torch.randn(4, 680)


def step(policy, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x, g = inputs()
    policy(x, g).square().sum().backward()
    optimizer.step()


def test_default_all_has_exact_forward_gradient_and_optimizer_parity():
    left = actor()
    right = deepcopy(left)
    configure_policy_update_scope(right, "all")
    a, b = torch.optim.Adam(left.parameters()), torch.optim.Adam(right.parameters())
    for _ in range(2):
        step(left, a)
        step(right, b)
    for (_, p), (_, q) in zip(left.named_parameters(), right.named_parameters(), strict=True):
        assert torch.equal(p, q) and torch.equal(p.grad, q.grad)
        assert a.state[p].keys() == b.state[q].keys()
        for key in a.state[p]:
            assert torch.equal(a.state[p][key], b.state[q][key])


@pytest.mark.parametrize("scope", ["decoder", "action-head"])
def test_frozen_parameters_and_adam_moments_stay_exact_but_input_derivatives_flow(scope):
    model = actor()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    step(model, opt)  # Seed all old Adam moments and leave nonzero stale gradients.
    old_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    old_opt = {name: deepcopy(opt.state[p]) for name, p in model.named_parameters()}
    old_groups = [id(p) for p in opt.param_groups[0]["params"]]
    x, g = inputs()
    baseline = model(x, g).detach().clone()
    configure_policy_update_scope(model, scope)
    assert torch.equal(model(x, g), baseline)
    assert old_groups == [id(p) for p in opt.param_groups[0]["params"]]
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen and all(p.grad is None for name, p in model.named_parameters() if name in frozen)
    x.requires_grad_()
    g.requires_grad_()
    model(x, g).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert g.grad is not None and torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0
    opt.step()
    changed = []
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(old_params[name], p) and p.grad is None
            for key in old_opt[name]:
                assert torch.equal(opt.state[p][key], old_opt[name][key])
        elif not torch.equal(old_params[name], p):
            changed.append(name)
    assert changed
    if scope == "action-head":
        last = str(len(model.decoder.mlp.net) - 1)
        assert set(changed) == {f"decoder.mlp.net.{last}.weight", f"decoder.mlp.net.{last}.bias"}
    # Old full Adam groups remain loadable and all parameters can be unfrozen.
    restored = actor()
    restored_opt = torch.optim.Adam(restored.parameters())
    restored_opt.load_state_dict(opt.state_dict())
    configure_policy_update_scope(model, "all")
    assert all(p.requires_grad for p in model.parameters())


def test_invalid_scope_does_not_partially_freeze_policy(tmp_path):
    model = actor()
    with pytest.raises(ValueError, match="Policy update scope"):
        configure_policy_update_scope(model, "head")
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="Policy update scope"):
        training_config("smoke", tmp_path, tmp_path, policy_update_scope="head")


def test_cli_new_legacy_resume_transfer_and_override_rules(tmp_path):
    assert parser().parse_args(["train"]).policy_update_scope is None
    assert (
        parser().parse_args(["train", "--policy-update-scope", "action-head"]).policy_update_scope
        == "action-head"
    )
    assert training_config("smoke", tmp_path, tmp_path).policy_trainer.update_scope == "all"
    saved = asdict(TrainConfig())
    path = tmp_path / "legacy.pt"
    saved["policy_trainer"].pop("update_scope")
    torch.save({"iteration": 1, "config": saved}, path)
    assert (
        training_config("smoke", tmp_path, tmp_path, resume=path).policy_trainer.update_scope
        == "all"
    )
    saved["policy_trainer"]["update_scope"] = "action-head"
    path = tmp_path / "saved.pt"
    torch.save({"iteration": 1, "config": saved}, path)
    assert (
        training_config("smoke", tmp_path, tmp_path, resume=path).policy_trainer.update_scope
        == "action-head"
    )
    assert (
        training_config(
            "smoke", tmp_path, tmp_path, resume=path, policy_update_scope="decoder"
        ).policy_trainer.update_scope
        == "decoder"
    )
    assert (
        training_config(
            "smoke", tmp_path, tmp_path, resume=path, transfer=True
        ).policy_trainer.update_scope
        == "all"
    )
    assert (
        training_config(
            "smoke", tmp_path, tmp_path, resume=path, transfer=True, policy_update_scope="decoder"
        ).policy_trainer.update_scope
        == "decoder"
    )
    saved["policy_trainer"]["update_scope"] = "corrupt"
    torch.save({"iteration": 1, "config": saved}, path)
    with pytest.raises(ValueError, match="Policy update scope"):
        training_config("smoke", tmp_path, tmp_path, resume=path)


@pytest.mark.parametrize("scope", ["action-head", None])
def test_export_records_saved_scope_and_legacy_default_without_changing_actor(tmp_path, scope):
    model = actor()
    checkpoint = tmp_path / "actor.pt"
    training = {} if scope is None else {"update_scope": scope}
    torch.save(
        {
            "iteration": 1,
            "policy": model.state_dict(),
            "config": {
                "policy": asdict(model.cfg),
                "env": {"future_frames": 10},
                "policy_trainer": training,
            },
        },
        checkpoint,
    )
    contract = export_policy(checkpoint, tmp_path / "export")
    assert contract["training_metadata"]["policy_update_scope"] == (scope or "all")


@pytest.mark.parametrize("scope", ["decoder", "action-head"])
def test_real_trainer_update_preserves_frozen_optimizer_steps_after_resume(monkeypatch, scope):
    from types import SimpleNamespace

    import supermimic.imagination.policy_trainer as module
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
    from supermimic.imagination.rollout import ImagRollout

    model = actor()
    old_opt = torch.optim.Adam(model.parameters())
    step(model, old_opt)
    wm = torch.nn.Linear(1, 1)
    wm.spec = SimpleNamespace(normalize_xi=lambda x: x, canonical=lambda x: x)
    cfg = PolicyTrainerCfg(
        horizon=2, batch_size=4, use_critic=False, compile=None, update_scope=scope, obs_noise=False
    )
    trainer = PolicyTrainer(
        wm, model, torch.nn.Linear(1, 1), SimpleNamespace(default_joint_pos=torch.zeros(29)), cfg
    )
    trainer.opt.load_state_dict(old_opt.state_dict())
    snapshots = {
        name: (p.detach().clone(), deepcopy(trainer.opt.state[p]))
        for name, p in model.named_parameters()
    }

    def rollout(wm, policy, lib, start, horizon, future_frames, **kwargs):
        x, g = inputs()
        actions = policy(x, g)
        reward = actions.square().mean(-1)
        return ImagRollout(
            states=torch.zeros(3, 4, 71),
            actions=actions[None].expand(2, -1, -1),
            rewards=reward[None].expand(2, -1),
            dhat=torch.zeros(3, 4),
            survival=torch.ones(3, 4),
            goals=torch.zeros(3, 4, 680),
            e_root=torch.zeros(3, 4, 4),
            frames=torch.zeros(3, 4, dtype=torch.long),
            reward_terms={},
        )

    monkeypatch.setattr(module, "imagine", rollout)
    buffer = SimpleNamespace(sample_starts=lambda *args: SimpleNamespace(xi=torch.zeros(4, 33)))
    metrics = trainer.update(buffer)
    assert torch.isfinite(torch.tensor(metrics["policy_grad_norm"]))
    for name, parameter in model.named_parameters():
        weight, state = snapshots[name]
        if not parameter.requires_grad:
            assert parameter.grad is None and torch.equal(weight, parameter)
            for key in state:
                assert torch.equal(state[key], trainer.opt.state[parameter][key])
        else:
            assert trainer.opt.state[parameter]["step"] == state["step"] + 1
