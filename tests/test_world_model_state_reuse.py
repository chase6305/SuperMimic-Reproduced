"""Reusing a root frame preserves the original dynamics composition and its adjoints."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.config import StateSpecCfg, WorldModelCfg
from supermimic.world_model.state_spec import StateSpec
from supermimic.world_model.world_model import WorldModel


@pytest.mark.parametrize("explicit", [False, True])
def test_dynamics_step_matches_independent_feature_and_integration_frames(explicit):
    torch.manual_seed(512)
    cfg = WorldModelCfg(hidden=(16, 16), state_spec=StateSpecCfg(explicit_integration=explicit))
    wm = WorldModel(cfg, SimpleNamespace(), "cpu").double()
    wm.spec = StateSpec(cfg.state_spec, "cpu", torch.float64)
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.01)
    state = torch.randn(5, 71, dtype=torch.float64)
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    state.requires_grad_()
    action = torch.randn(5, 29, dtype=torch.float64, requires_grad=True)
    context = torch.randn(5, 33, dtype=torch.float64, requires_grad=True)

    def independent(s, a, xi):
        delta = wm.dynamics(wm.spec.canonical(s), a, wm.spec.normalize_xi(xi))
        return wm.spec.integrate(s, delta)

    actual, expected = wm.step(state, action, context), independent(state, action, context)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    inputs = (state, action, context, *wm.dynamics.parameters())
    actual_gradients = torch.autograd.grad((actual * probe).sum(), inputs)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10)
    # A second state must use its own frame, with no persistent per-model cache.
    following = state.detach().clone()
    following[:, 3:7] = torch.randn(5, 4, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(
            wm.step(following, action, context),
            independent(following, action, context),
            atol=0,
            rtol=0,
        )
