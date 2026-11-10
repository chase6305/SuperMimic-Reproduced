"""Optional dense imitation penalties through the learned dynamics.

These auxiliary objectives leave the baseline analytic evaluation reward intact.
"""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import quat_apply, quat_conjugate, quat_mul, rotvec_from_quat


def tracking_auxiliary(
    predicted: torch.Tensor,
    target: torch.Tensor,
    survival: torch.Tensor,
    weights: tuple[float, float, float, float],
    min_survival: float = 1e-6,
    dt: float = 0.02,
    orientation_mode: str = "chordal",
    joint_weights: tuple[float, ...] | None = None,
    angular_velocity_weight: float = 0.0,
    root_xy_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights: joint pose, root height, root orientation, vertical velocity.

    Targets correspond to the command used for reaching each predicted state.
    Detach survival weights so ending a branch cannot reduce its imitation penalty.
    """
    alive = survival.detach() > min_survival
    weight = torch.where(alive, survival.detach(), torch.zeros_like(survival))
    p = torch.where(alive[..., None], predicted, torch.zeros_like(predicted))
    t = torch.where(alive[..., None], target, torch.zeros_like(target))
    pq = p[..., 3:7] / p[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tq = t[..., 3:7] / t[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    identity = pq.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(pq)
    pq = torch.where(alive[..., None], pq, identity)
    tq = torch.where(alive[..., None], tq, identity)
    if orientation_mode == "geodesic":
        rotation_error = rotvec_from_quat(quat_mul(quat_conjugate(tq), pq)).square().sum(-1)
    elif orientation_mode == "chordal":
        rotation_error = (1 - (pq * tq).sum(-1).square()).clamp_min(0)
    else:
        raise ValueError("Unknown orientation loss")
    squared_joints = (p[..., 7:36] - t[..., 7:36]).square()
    if joint_weights is None:
        joint_error = squared_joints.mean(-1)
    else:
        jw = p.new_tensor(joint_weights)
        joint_error = (squared_joints * jw).sum(-1) / jw.sum()
    errors = torch.stack(
        [
            joint_error,
            (p[..., 2] - t[..., 2]).square(),
            rotation_error,
            (p[..., 38] - t[..., 38]).square(),
        ],
        -1,
    )
    if angular_velocity_weight:
        # Learned state stores body-frame omega; the reference stores world omega.
        angular_error = (quat_apply(pq, p[..., 39:42]) - t[..., 39:42]).square().mean(-1)
        errors = torch.cat([errors, angular_error[..., None]], -1)
        weights = (*weights, angular_velocity_weight)
    if root_xy_weight:
        # World XY displacement, matching the norm used by trajectory evaluation.
        # Shared translations cancel; survival is detached above, as for all terms.
        xy_error = (p[..., :2] - t[..., :2]).square().sum(-1)
        errors = torch.cat([errors, xy_error[..., None]], -1)
        weights = (*weights, root_xy_weight)
    terms = (weight[..., None] * errors).sum(
        dim=tuple(range(errors.ndim - 1))
    ) / weight.sum().clamp_min(1)
    return dt * (terms * terms.new_tensor(weights)).sum(), terms
