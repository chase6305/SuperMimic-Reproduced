"""Parallel affine composition must preserve the soft-terminal lambda return."""

import pytest
import torch

from supermimic.imagination.objective import _gae_targets_scan, gae_targets


@torch.no_grad()
def sequential(rewards, termination, values, gamma, lam):
    running = values[-1].clone()
    result = torch.empty_like(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * (1 - termination[t]) * (
            (1 - lam) * values[t + 1] + lam * running
        )
        result[t] = running
    return result


@pytest.mark.parametrize("horizon", [0, 1, 3, 8, 31, 32, 64])
def test_affine_scan_matches_double_precision_returns_without_mutation(horizon):
    torch.manual_seed(700 + horizon)
    for gamma, lam in [(0.0, 0.95), (0.99, 0.0), (0.99, 0.95), (1.0, 1.0)]:
        for layout in ("soft", "never", "always"):
            rewards = (torch.randn(horizon, 5, dtype=torch.float64) * 0.1).requires_grad_()
            values = (torch.randn(horizon + 1, 5, dtype=torch.float64) * 0.2).requires_grad_()
            termination = torch.rand_like(rewards)
            if layout == "never":
                termination.zero_()
            elif layout == "always":
                termination.fill_(1)
            else:
                termination[::3, ::2] = 1
            inputs = (rewards, termination, values)
            saved = [value.detach().clone() for value in inputs]
            rng = torch.get_rng_state()
            expected = sequential(*inputs, gamma, lam)
            actual = _gae_targets_scan(*inputs, gamma, lam)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            # CPU calls retain the original evaluation order.
            torch.testing.assert_close(gae_targets(*inputs, gamma, lam), expected, atol=0, rtol=0)
            assert not actual.requires_grad
            assert actual.dtype == rewards.dtype and actual.shape == rewards.shape
            assert torch.equal(torch.get_rng_state(), rng)
            for value, original in zip(inputs, saved, strict=True):
                torch.testing.assert_close(value, original, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for target dispatch")
@pytest.mark.parametrize("horizon,batch", [(8, 64), (32, 64), (64, 512)])
def test_cuda_targets_and_other_dtype_fallbacks(horizon, batch):
    torch.manual_seed(900 + horizon)
    for dtype_case in ("float32", "float16", "mixed"):
        dtype = torch.float16 if dtype_case == "float16" else torch.float32
        rewards = torch.randn(horizon, batch, device="cuda", dtype=dtype) * 0.1
        values = torch.randn(horizon + 1, batch, device="cuda", dtype=dtype) * 0.2
        if dtype_case == "mixed":
            values = values.double()
        for gamma, lam in [(0.99, 0.0), (0.99, 0.95), (1.0, 1.0)]:
            for terminal in (False, True):
                termination = torch.rand_like(rewards) if terminal else torch.zeros_like(rewards)
                if terminal:
                    termination[::3, ::2] = 1
                rng = torch.cuda.get_rng_state()
                expected = sequential(rewards, termination, values, gamma, lam)
                actual = gae_targets(rewards, termination, values, gamma, lam)
                tolerance = 2e-6 if dtype_case == "float32" else 0
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                assert not actual.requires_grad
                assert actual.dtype == rewards.dtype and actual.shape == rewards.shape
                assert torch.equal(torch.cuda.get_rng_state(), rng)
