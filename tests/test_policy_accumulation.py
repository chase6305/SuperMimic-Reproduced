"""One actor step must average microbatches without retaining their graphs."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import supermimic.imagination.policy_trainer as module
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import ImagRollout


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, feature, goal, context, root_error):
        return self.bias + feature[..., 0] * 0.1


def trainer(accumulation, use_critic=False):
    result = PolicyTrainer.__new__(PolicyTrainer)
    result.cfg = PolicyTrainerCfg(
        horizon=2,
        batch_size=4 // accumulation,
        gradient_accumulation_steps=accumulation,
        use_critic=use_critic,
        compile=None,
        obs_noise=False,
        critic_epochs=1,
        critic_minibatches=1,
        grad_clip_norm=1000,
        grad_clip_value=1000,
    )
    result.wm = torch.nn.Linear(1, 1)
    result.wm.spec = SimpleNamespace(normalize_xi=lambda x: x, canonical=lambda x: x)
    result.policy = torch.nn.Linear(1, 1, bias=False)
    result.policy.weight.data.fill_(0.2)
    result.critic = Value()
    result.opt = torch.optim.SGD(result.policy.parameters(), lr=0.1)
    result.critic_opt = torch.optim.SGD(result.critic.parameters(), lr=0.1)
    result.lib = result.step_fn = result.foot_support = None
    return result


class Buffer:
    def __init__(self):
        self.next = 0

    def sample_starts(self, batch_size, history):
        inputs = torch.arange(self.next + 1, self.next + batch_size + 1).float()[:, None]
        self.next += batch_size
        return SimpleNamespace(xi=torch.zeros(batch_size, 33), inputs=inputs)


def synthetic_rollout(wm, policy, lib, start, horizon, future_frames, **kwargs):
    n = len(start.inputs)
    action = policy(start.inputs).flatten()
    states = torch.zeros(horizon + 1, n, 71)
    states[..., 0] = action
    states[..., 3] = 1
    return ImagRollout(
        states=states,
        actions=action[None, :, None].expand(horizon, n, 29),
        rewards=action[None].expand(horizon, n),
        dhat=torch.zeros(horizon + 1, n),
        survival=torch.ones(horizon + 1, n),
        goals=torch.zeros(horizon + 1, n, 2),
        e_root=torch.zeros(horizon + 1, n, 4),
        frames=torch.zeros(horizon + 1, n, dtype=torch.long),
        reward_terms={},
    )


def test_accumulation_matches_analytic_full_batch_gradient_and_one_step(monkeypatch):
    monkeypatch.setattr(module, "imagine", synthetic_rollout)
    full, micro = trainer(1), trainer(2)
    steps = []
    original_step = micro.opt.step

    def step():
        steps.append(True)
        return original_step()

    monkeypatch.setattr(micro.opt, "step", step)
    a = full.update(Buffer())
    b = micro.update(Buffer())
    # d[-mean_n(sum_t gamma**t * reward) / H]/dw = -2.5*(1+.99)/2.
    expected_gradient = -2.5 * (1 + 0.99) / 2
    torch.testing.assert_close(micro.policy.weight.grad, torch.tensor([[expected_gradient]]))
    torch.testing.assert_close(micro.policy.weight, torch.tensor([[0.2 - 0.1 * expected_gradient]]))
    torch.testing.assert_close(full.policy.weight, micro.policy.weight)
    assert len(steps) == 1 and b["actor_microbatches"] == 2
    assert a["imag_objective"] == pytest.approx(b["imag_objective"])
    assert all(p.requires_grad for p in micro.wm.parameters())
    assert all(p.grad is None for p in micro.wm.parameters())


def test_critic_cache_is_detached_and_value_fit_runs_after_actor_step(monkeypatch):
    monkeypatch.setattr(module, "imagine", synthetic_rollout)
    model = trainer(2, use_critic=True)
    steps, caches = [], []
    original_step, original_fit = model.opt.step, model._fit_critic

    def step():
        steps.append(True)
        return original_step()

    def fit(batch):
        assert len(steps) == 1
        for field in fields(batch):
            value = getattr(batch, field.name)
            assert not value.requires_grad and value.grad_fn is None
        caches.append(batch)
        return original_fit(batch)

    monkeypatch.setattr(model.opt, "step", step)
    monkeypatch.setattr(model, "_fit_critic", fit)
    before = model.critic.bias.detach().clone()
    metrics = model.update(Buffer())
    assert len(caches) == 2 and metrics["critic_loss"] >= 0
    assert model.critic.bias.detach() != before
    assert all(p.requires_grad for p in model.critic.parameters())


def test_failed_microbatch_cannot_apply_partial_actor_update(monkeypatch):
    model = trainer(2)
    before = model.policy.weight.detach().clone()
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise FloatingPointError("Invalid second microbatch")
        return synthetic_rollout(*args, **kwargs)

    monkeypatch.setattr(module, "imagine", broken)
    with pytest.raises(FloatingPointError, match="second microbatch"):
        model.update(Buffer())
    torch.testing.assert_close(model.policy.weight, before)
    assert all(p.requires_grad for p in model.wm.parameters())
    assert all(p.requires_grad for p in model.critic.parameters())


def test_accumulation_count_rejects_invalid_work_budgets(tmp_path):
    from supermimic.profiles import training_config

    for count in (0, -1, 17, 1.5):
        with pytest.raises(ValueError, match="gradient accumulation"):
            PolicyTrainer(
                None, None, None, None, PolicyTrainerCfg(gradient_accumulation_steps=count)
            )
        with pytest.raises(ValueError, match="gradient accumulation"):
            training_config("smoke", tmp_path, tmp_path / "run", policy_gradient_accumulation=count)
