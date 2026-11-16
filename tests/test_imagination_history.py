"""Batching initial sensor computations preserves history, RNG and BPTT inputs."""

from types import SimpleNamespace

import pytest
import torch
from test_imagination_stability import ToyLibrary

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.rollout import OBS_NOISE_HALF, imagine
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


@pytest.mark.parametrize("history_length", [1, 10])
@pytest.mark.parametrize("noise", [False, True])
def test_initial_history_matches_individual_frames_and_preserves_noise_rng(history_length, noise):
    torch.manual_seed(101)
    n = 3
    states = torch.randn(history_length, n, 71)
    states[..., 3:7] /= states[..., 3:7].norm(dim=-1, keepdim=True)
    states.requires_grad_()
    actions = torch.randn(history_length, n, 29, requires_grad=True)
    context = torch.randn(n, 33, requires_grad=True)
    start = StartBatch(
        states[-1], context, torch.zeros(n, dtype=torch.long), actions[-1], states, actions
    )
    library = ToyLibrary()
    layer = torch.nn.Linear(history_length * 93, 29)
    inputs = (states, actions, context, *layer.parameters())
    expected_frames = []
    torch.manual_seed(207)
    for i in range(history_length):
        observation = analytic_observation(
            states[i], actions[i], context, library.default_joint_pos
        )
        if noise:
            observation = observation + (torch.rand_like(observation) * 2 - 1) * OBS_NOISE_HALF
        expected_frames.append(observation)
    expected_rng = torch.get_rng_state()
    expected_history = flatten_history_term_major(torch.stack(expected_frames))
    expected_action = layer(expected_history)
    expected_gradients = torch.autograd.grad(expected_action.square().sum(), inputs)
    supplied = []

    class Policy(torch.nn.Module):
        def forward(self, history, goal):
            supplied.append((history.detach(), torch.get_rng_state()))
            return layer(history)

    def step(state, action, previous, xi, ref):
        reward = action[:, 0]
        return (
            state,
            reward,
            reward[None].expand(8, n),
            torch.zeros(n),
            analytic_observation(state, action, xi, library.default_joint_pos),
            state[:, :7],
        )

    heads = SimpleNamespace(anchor_index=0, root_error_features=lambda s, r: torch.zeros(n, 4))
    torch.manual_seed(207)
    rollout = imagine(
        SimpleNamespace(heads=heads),
        Policy(),
        library,
        start,
        horizon=1,
        obs_noise=noise,
        step_fn=step,
    )
    actual_gradients = torch.autograd.grad(rollout.actions[0].square().sum(), inputs)
    torch.testing.assert_close(supplied[0][0], expected_history, rtol=0, atol=0)
    assert torch.equal(supplied[0][1], expected_rng)
    torch.testing.assert_close(rollout.actions[0], expected_action, rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
