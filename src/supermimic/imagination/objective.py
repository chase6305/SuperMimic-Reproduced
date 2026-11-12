"""Policy objective (Eq. 2) and critic GAE targets (Eq. 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from supermimic.imagination.rollout import ImagRollout


def survival_weighted_mse(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Fit values under the probability of reaching each imagined state.

    Mask before subtraction/squaring so an unreachable infinite target cannot
    contaminate a zero-weight term. Scale before squaring to reduce overflow.
    """
    active = weight > 0
    safe_prediction = torch.where(active, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(active, target, torch.zeros_like(target))
    residual = (safe_prediction - safe_target) * weight.sqrt()
    return residual.square().sum() / weight.sum().clamp_min(1e-8)


def policy_objective(
    ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float
) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H ] (SHAC-style; V_H keeps the gradient w.r.t. s_H).

    terminal_values: (n,) critic estimate at s_H, or None for the critic-free variant.
    """
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + (gamma**H) * ro.survival[-1] * terminal_values
    return J.mean()


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out
