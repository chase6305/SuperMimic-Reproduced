"""Optional support loss through actual G1 foot collision geometry and FK.

This is a training objective; physical contacts and acceptance use solved forces.
"""

from __future__ import annotations

import math
import re

import mujoco
import torch

from supermimic.envs.mdp.rewards import reanchor_reference
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply, quat_mul


class FootSupport:
    def __init__(self, model: mujoco.MjModel, fk: G1Kinematics):
        self.fk = fk
        self.anchor = fk.body_index("torso_link")
        self.effectors = fk.body_indices(
            (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
        )
        ids, groups = [], [[], []]
        for index in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").split("/")[-1]
            match = re.fullmatch(r"(left|right)_foot[1-7]_collision", name)
            if match is None or not model.geom_contype[index]:
                continue
            if int(model.geom_type[index]) not in (
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_SPHERE),
            ):
                raise ValueError("Foot support requires capsule or sphere foot colliders")
            groups[0 if match[1] == "left" else 1].append(len(ids))
            ids.append(index)
        if not all(groups):
            raise ValueError("Both feet must have physical collision geometry")

        def tensor(value, **kwargs):
            return torch.as_tensor(value, device=fk.device, **kwargs)

        self.groups = [tensor(group, dtype=torch.long) for group in groups]
        # Training scene names include robot/ and may contain additional bodies.
        # Map by robot body names rather than assuming matching model body IDs.
        bodies = []
        for index in ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index]))
            bodies.append(fk.body_index(name.split("/")[-1]))
        self.body = tensor(bodies, dtype=torch.long)
        self.position = tensor(model.geom_pos[ids].copy(), dtype=fk.dtype)
        self.quaternion = tensor(model.geom_quat[ids].copy(), dtype=fk.dtype)
        self.radius = tensor(model.geom_size[ids, 0].copy(), dtype=fk.dtype)
        self.half_length = tensor(
            [
                model.geom_size[index, 1]
                if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE
                else 0
                for index in ids
            ],
            dtype=fk.dtype,
        )
        self.up = tensor([0, 0, 1], dtype=fk.dtype)

    def soles(self, bodies: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Minimum collider height and its kinematic normal velocity, for each foot."""
        quat = bodies["quat"].index_select(-2, self.body)
        pos = bodies["pos"].index_select(-2, self.body)
        offset = quat_apply(quat, self.position.expand(*quat.shape[:-1], 3))
        orientation = quat_mul(quat, self.quaternion.expand_as(quat))
        axis = quat_apply(orientation, self.up.expand(*quat.shape[:-1], 3))
        height = (pos + offset)[..., 2] - self.radius - self.half_length * axis[..., 2].abs()
        normal_velocity = None
        if "lin_vel" in bodies:
            omega = bodies["ang_vel"].index_select(-2, self.body)
            center_velocity = bodies["lin_vel"].index_select(-2, self.body) + torch.cross(
                omega, offset, dim=-1
            )
            extent_velocity = (
                self.half_length * axis[..., 2].sign() * torch.cross(omega, axis, dim=-1)[..., 2]
            )
            normal_velocity = center_velocity[..., 2] - extent_velocity
        heights, velocities = [], []
        for group in self.groups:
            minimum, which = height.index_select(-1, group).min(-1)
            heights.append(minimum)
            if normal_velocity is not None:
                velocities.append(
                    normal_velocity.index_select(-1, group).gather(-1, which[..., None]).squeeze(-1)
                )
        return torch.stack(heights, -1), torch.stack(velocities, -1) if velocities else None

    def loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        survival: torch.Tensor,
        height_weight: float,
        velocity_weight: float,
        *,
        min_survival: float = 1e-6,
        dt: float = 0.02,
        effector_position_weight: float = 0.0,
        effector_position_weights: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if effector_position_weights is not None and (
            len(effector_position_weights) != 4
            or any(not math.isfinite(v) or v <= 0 for v in effector_position_weights)
        ):
            raise ValueError("Effector position weights must contain four finite positive values")
        alive = survival.detach() > min_survival
        identity = predicted.new_tensor([1, 0, 0, 0]).expand(*predicted.shape[:-1], 4)

        def sanitize(state):
            safe = torch.where(alive[..., None], state, torch.zeros_like(state))
            return torch.cat(
                [
                    safe[..., :3],
                    torch.where(alive[..., None], safe[..., 3:7], identity),
                    safe[..., 7:],
                ],
                -1,
            )

        p, t = sanitize(predicted), sanitize(target)
        predicted_bodies = self.fk(p[..., :36], p[..., 36:])
        heights, velocity = self.soles(predicted_bodies)
        with torch.no_grad():
            reference_bodies = self.fk(t[..., :36])
            reference_height, _ = self.soles(reference_bodies)
            stance = alive[..., None] & (reference_height <= 0.03)
            weight = survival.detach()[..., None] * stance
        denominator = weight.sum().clamp_min(1)
        terms = torch.stack(
            [
                (weight * heights.square()).sum() / denominator,
                (weight * velocity.square()).sum() / denominator,
            ]
        )
        coefficients = [height_weight, velocity_weight]
        if effector_position_weight:
            ai, ei = self.anchor, self.effectors
            # Same torso xy/yaw reanchoring as the fixed physical evaluation.
            # Retain gradients through the actual anchor; references are fixed.
            relative, _ = reanchor_reference(
                reference_bodies["pos"][..., ei, :],
                reference_bodies["quat"][..., ei, :],
                reference_bodies["pos"][..., ai, :],
                reference_bodies["quat"][..., ai, :],
                predicted_bodies["pos"][..., ai, :],
                predicted_bodies["quat"][..., ai, :],
            )
            squared = (predicted_bodies["pos"][..., ei, :] - relative).square().sum(-1)
            if effector_position_weights is None or tuple(effector_position_weights) == (1.0,) * 4:
                squared = squared.mean(-1)
            else:
                limb_weight = squared.new_tensor(effector_position_weights)
                squared = (squared * limb_weight).sum(-1) / limb_weight.sum()
            reached_weight = torch.where(alive, survival.detach(), torch.zeros_like(survival))
            position_mse = (reached_weight * squared).sum() / reached_weight.sum().clamp_min(1)
            terms = torch.cat([terms, position_mse[None]])
            coefficients.append(effector_position_weight)
        loss = dt * (terms * terms.new_tensor(coefficients)).sum()
        return loss, terms
