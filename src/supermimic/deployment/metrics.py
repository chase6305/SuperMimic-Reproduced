"""Tracking metrics from actual saved states, using the reproduction's analytic rewards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.deployment.core import heading_error
from supermimic.envs.mdp.rewards import reanchor_reference
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.eval.quality import motion_quality, worst_window_rmse
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.reference import ReferenceFrame


def airborne_metrics(physical_contacts: np.ndarray, physics_dt: float) -> dict:
    """Observed unloaded intervals; these alone do not establish successful jumps."""
    if physics_dt <= 0 or not np.isfinite(physics_dt):
        raise ValueError("Physics timestep must be finite and positive")
    physical = np.asarray(physical_contacts)
    if physical.ndim != 2 or physical.shape[1] != 13:
        raise ValueError("Expected physical contact records with 13 columns")
    foot_load = np.linalg.norm(physical[:, 1:7].reshape(-1, 2, 3), axis=-1)
    unloaded = (foot_load < 1.0).all(-1) & (physical[:, 10] == 0)
    changes = np.diff(np.concatenate([[False], unloaded, [False]]).astype(np.int8))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    durations = lengths * physics_dt
    return {
        "airborne_substeps": int(unloaded.sum()),
        "airborne_time_s": float(unloaded.sum() * physics_dt),
        "longest_airborne_s": float(durations.max()) if len(durations) else 0.0,
        "airborne_intervals_ge_40ms": int(np.count_nonzero(durations >= 0.04)),
        "airborne_definition": "Each foot solved-force norm below 1 N, no nonfoot ground force; sampled every physics substep. Unloaded intervals are not a jump-success metric.",
    }


@torch.inference_mode()
def tracking_metrics(
    states: np.ndarray, actions: np.ndarray, frames: np.ndarray, motion: Path, contract: dict
) -> dict:
    torch.set_num_threads(1)
    model = load_g1_model()
    fk = G1Kinematics(model, "cpu")
    limits = model.jnt_range[1:].copy()
    # Same 0.9 soft-limit interval as the training Entity.
    center = limits.mean(axis=1)
    half = (limits[:, 1] - limits[:, 0]) * 0.45
    soft = torch.tensor(np.stack([center - half, center + half], axis=-1), dtype=torch.float32)
    default_q = torch.tensor(contract["default_joint_pos"], dtype=torch.float32)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, soft, default_q)
    with np.load(motion, allow_pickle=False) as data:
        names = data["body_names"].tolist()
        body_indices = [names.index(name) for name in TRACKED_BODIES]
        root_index = names.index(ROOT_BODY)
        reference = {
            key: data[key].copy()
            for key in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "joint_pos",
            )
        }
        expected_steps = len(data["joint_pos"]) - 1
    totals = {}
    xy_sq = joint_sq = velocity_sq = yaw_sq = body_sq = 0.0
    yaw_max = 0.0
    actual_speed_sum = reference_speed_sum = 0.0
    xy_max = 0.0
    n = len(actions)
    body_errors = []
    previous_actions = np.concatenate([np.zeros((1, 29), dtype=np.float32), actions[:-1]])
    for begin in range(0, n, 128):
        end = min(begin + 128, n)
        indices = frames[begin:end]
        current = torch.tensor(states[begin + 1 : end + 1], dtype=torch.float32)
        action = torch.tensor(actions[begin:end])
        previous = torch.tensor(previous_actions[begin:end])
        ref = ReferenceFrame(
            *[
                torch.tensor(reference[key][indices][:, body_indices], dtype=torch.float32)
                for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
            ],
            torch.tensor(reference["body_pos_w"][indices, root_index], dtype=torch.float32),
            torch.tensor(reference["body_quat_w"][indices, root_index], dtype=torch.float32),
        )
        bodies = heads.robot_bodies(current)
        terms = heads.reward(current, action, previous, ref, bodies)
        ai = heads.anchor_index
        relative, _ = reanchor_reference(
            ref.body_pos_w,
            ref.body_quat_w,
            ref.body_pos_w[:, ai],
            ref.body_quat_w[:, ai],
            bodies["pos"][:, ai],
            bodies["quat"][:, ai],
        )
        per_body = (relative - bodies["pos"]).square().sum(-1)
        body_sq += float(per_body.sum())
        body_errors.append(per_body.numpy())
        for key, values in terms.items():
            totals[key] = totals.get(key, 0.0) + float(values.sum())
        xy = np.linalg.norm(
            states[begin + 1 : end + 1, :2] - reference["body_pos_w"][indices, root_index, :2],
            axis=-1,
        )
        xy_sq += float(np.square(xy).sum())
        xy_max = max(xy_max, float(xy.max()))
        yaw = heading_error(
            states[begin + 1 : end + 1, 3:7], reference["body_quat_w"][indices, root_index]
        )
        yaw_sq += float(np.square(yaw).sum())
        yaw_max = max(yaw_max, float(np.abs(yaw).max()))
        joint_sq += float(
            np.square(states[begin + 1 : end + 1, 7:36] - reference["joint_pos"][indices]).sum()
        )
        actual_velocity = states[begin + 1 : end + 1, 36:38]
        reference_velocity = reference["body_lin_vel_w"][indices, root_index, :2]
        velocity_sq += float(np.square(actual_velocity - reference_velocity).sum())
        actual_speed_sum += float(np.linalg.norm(actual_velocity, axis=-1).sum())
        reference_speed_sum += float(np.linalg.norm(reference_velocity, axis=-1).sum())
    return {
        "quality": motion_quality(states, frames, reference, root_index, contract["control_dt"]),
        "normalized_return_executed_steps": totals.get("total", 0.0) / (0.1 * max(n, 1)),
        "normalized_return_full_clip_zero_padded": totals.get("total", 0.0)
        / (0.1 * max(expected_steps, n, 1)),
        "clip_coverage": min(n / max(expected_steps, 1), 1.0),
        "root_xy_rmse_m": (xy_sq / max(n, 1)) ** 0.5,
        "root_xy_max_error_m": xy_max,
        "root_yaw_rmse_rad": (yaw_sq / max(n, 1)) ** 0.5,
        "root_yaw_max_error_rad": yaw_max,
        "joint_rmse_rad": (joint_sq / max(n * 29, 1)) ** 0.5,
        "body_relative_pos_rmse_m": (body_sq / max(n * len(TRACKED_BODIES), 1)) ** 0.5,
        "body_relative_worst_1s_rmse_m": dict(
            zip(
                TRACKED_BODIES,
                worst_window_rmse(
                    np.concatenate(body_errors), max(1, round(1 / contract["control_dt"]))
                ).tolist(),
                strict=True,
            )
        )
        if body_errors
        else {},
        "root_xy_velocity_rmse_mps": (velocity_sq / max(n, 1)) ** 0.5,
        "mean_root_xy_speed_mps": actual_speed_sum / max(n, 1),
        "mean_reference_xy_speed_mps": reference_speed_sum / max(n, 1),
        "root_xy_distance_m": float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=-1).sum()),
        "mean_reward_terms": {key: value / max(n, 1) for key, value in totals.items()},
        "reward_protocol": "Cuboi analytic reward on reached state and current target frame; dt=0.02",
    }
