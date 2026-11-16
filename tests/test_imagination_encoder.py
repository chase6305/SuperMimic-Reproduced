"""Batched motion encoding retains feedback, clipping and frozen-layer derivatives."""

from types import SimpleNamespace

import pytest
import torch
from test_imagination_stability import ToyLibrary

from supermimic.config import PolicyCfg
from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.policy_trainer import configure_policy_update_scope
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import analytic_observation


@pytest.mark.parametrize("scope", ["all", "decoder", "action-head"])
@pytest.mark.parametrize("clip", [None, 0.001])
@pytest.mark.parametrize("anchor", [False, True])
def test_batched_encoding_matches_per_step_actor_and_feedback_gradients(scope, clip, anchor):
    torch.manual_seed(612)
    n, history, horizon, future = 3, 3, 4, 2
    actor = Policy(
        PolicyCfg(
            history=history, goal_dim=future * 68 + 9 * anchor,
            encoder_hidden=(16,), decoder_hidden=(16,), action_clip=clip,
        )
    ).double()
    configure_policy_update_scope(actor, scope)

    class Motion(ToyLibrary):
        def goal_window(self, frame, future_frames):
            targets = (frame[:, None] + torch.arange(future_frames)).clamp_max(1000)
            features = torch.arange(68, dtype=torch.float64) * 0.01
            return torch.sin(targets[..., None] * 0.1 + features).flatten(1)

    library = Motion()
    states = torch.randn(history, n, 71, dtype=torch.float64) * 0.1
    states[..., 3:7] /= states[..., 3:7].norm(dim=-1, keepdim=True)
    states.requires_grad_()
    actions = (torch.randn(history, n, 29, dtype=torch.float64) * 0.01).requires_grad_()
    context = torch.randn(n, 33, dtype=torch.float64, requires_grad=True)
    start = StartBatch(
        states[-1], context, torch.tensor([0, 996, 999]), actions[-1], states, actions
    )
    heads = SimpleNamespace(
        anchor_index=0,
        root_error_features=lambda s, ref: s[:, :4],
        robot_bodies=lambda s: {"pos": s[:, None, :3], "quat": s[:, None, 3:7]},
    )

    def step(state, action, previous, xi, reference):
        following = state.clone()
        following[:, 7:36] = state[:, 7:36] + 0.02 * action
        following[:, 42:] = state[:, 42:] * 0.8 + 0.2 * action
        following[:, 2] = state[:, 2] + 0.01 * action.mean(-1)
        reward = (
            1 - following[:, 7:36].square().mean(-1)
            - (action - previous).square().mean(-1)
        )
        return (
            following,
            reward,
            reward[None].expand(8, n),
            torch.tensor([-100.0, 100.0, -100.0], dtype=torch.float64),
            analytic_observation(following, action, xi, library.default_joint_pos),
            following[:, :7],
        )

    class PerStepPolicy(torch.nn.Module):
        def forward(self, observation, goal):
            return actor(observation, goal)

    inputs = (states, actions, context, *[p for p in actor.parameters() if p.requires_grad])

    def run(policy):
        torch.manual_seed(733)
        rollout = imagine(
            SimpleNamespace(heads=heads), policy, library, start, horizon,
            future_frames=future, obs_noise=True, anchor_obs=anchor, step_fn=step,
        )
        terminal_value = rollout.states[-1, :, 7:36].mean(-1)
        gradients = torch.autograd.grad(policy_objective(rollout, terminal_value, 0.99), inputs)
        return rollout, gradients, torch.get_rng_state()

    expected, expected_gradients, expected_rng = run(PerStepPolicy())
    actual, actual_gradients, actual_rng = run(actor)
    assert torch.equal(actual_rng, expected_rng)
    assert actual.survival[-1, 1] == 0  # Exercise absorbing branches and their goal substitution.
    for name in ("states", "actions", "rewards", "dhat", "survival", "goals", "e_root"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=1e-12, rtol=1e-12
        )
    for observed, target in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, target, atol=1e-10, rtol=1e-10)
    if clip is not None:
        assert actual.actions.abs().max() <= clip
