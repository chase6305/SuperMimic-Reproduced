"""Pose and motion fidelity from actual states, beyond merely staying upright."""

from __future__ import annotations

import numpy as np


def worst_window_rmse(squared_errors: np.ndarray, window: int) -> np.ndarray:
    """Maximum rolling RMS over time, preserving trailing signal dimensions."""
    values = np.asarray(squared_errors, dtype=np.float64)
    count = min(window, len(values))
    if count < 1:
        raise ValueError("A positive window and nonempty errors are required")
    cumulative = np.concatenate(
        [np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)], axis=0
    )
    means = (cumulative[count:] - cumulative[:-count]) / count
    return np.sqrt(np.maximum(means, 0).max(axis=0))


def motion_quality(
    states: np.ndarray,
    frames: np.ndarray,
    reference: dict,
    root_index: int,
    control_dt: float = 0.02,
) -> dict:
    n = len(frames)
    if not n:
        return {"samples": 0, "joint_motion_correlation": 0.0}
    actual = np.asarray(states[1 : n + 1], dtype=np.float64)
    target = np.asarray(reference["joint_pos"][frames], dtype=np.float64)
    error = actual[:, 7:36] - target
    actual_centered = actual[:, 7:36] - actual[:, 7:36].mean(0)
    target_centered = target - target.mean(0)
    active = target.std(0) >= 0.1
    covariance = (actual_centered * target_centered).mean(0)
    denominator = actual[:, 7:36].std(0) * target.std(0)
    correlation = np.divide(covariance, denominator, out=np.zeros(29), where=denominator > 1e-8)
    root_position = reference["body_pos_w"][frames, root_index]
    root_orientation = reference["body_quat_w"][frames, root_index].astype(np.float64)
    actual_orientation = actual[:, 3:7]
    root_orientation /= np.linalg.norm(root_orientation, axis=-1, keepdims=True)
    actual_orientation = actual_orientation / np.linalg.norm(
        actual_orientation, axis=-1, keepdims=True
    )
    angle = 2 * np.arccos(np.clip(np.abs((actual_orientation * root_orientation).sum(-1)), 0, 1))
    height_error = actual[:, 2] - root_position[:, 2]
    vertical_error = actual[:, 38] - reference["body_lin_vel_w"][frames, root_index, 2]
    amplitude = {}
    for name, sl in (("legs", slice(0, 12)), ("waist", slice(12, 15)), ("arms", slice(15, 29))):
        use = active[sl]
        numerator = np.square(actual[:, 7:36].std(0)[sl][use]).sum()
        denominator = np.square(target.std(0)[sl][use]).sum()
        amplitude[name] = float(np.sqrt(numerator / denominator)) if denominator > 1e-8 else None
    reference_gravity_z = -(1 - 2 * np.square(root_orientation[:, 1:3]).sum(-1))
    actual_gravity_z = -(1 - 2 * np.square(actual_orientation[:, 1:3]).sum(-1))
    inverted = reference_gravity_z > 0
    inversion = {"reference_frames": int(inverted.sum())}
    if inverted.any():
        inversion.update(
            recall=float((actual_gravity_z[inverted] > 0).mean()),
            root_orientation_rmse_rad=float(np.sqrt(np.mean(angle[inverted] ** 2))),
            root_height_rmse_m=float(np.sqrt(np.mean(height_error[inverted] ** 2))),
            joint_rmse_rad=float(np.sqrt(np.mean(error[inverted] ** 2))),
        )
    if control_dt <= 0 or not np.isfinite(control_dt):
        raise ValueError("Invalid control timestep")
    window = max(1, round(1 / control_dt))
    return {
        "samples": n,
        "root_orientation_rmse_rad": float(np.sqrt(np.mean(angle**2))),
        "root_height_rmse_m": float(np.sqrt(np.mean(height_error**2))),
        "root_vertical_velocity_rmse_mps": float(np.sqrt(np.mean(vertical_error**2))),
        "worst_1s_joint_rmse_rad": float(worst_window_rmse(np.square(error).mean(-1), window)),
        "worst_1s_root_orientation_rmse_rad": float(worst_window_rmse(angle**2, window)),
        "worst_1s_root_height_rmse_m": float(worst_window_rmse(height_error**2, window)),
        "joint_amplitude_ratio_by_group": amplitude,
        "inversion": inversion,
        "joint_group_rmse_rad": {
            name: float(np.sqrt(np.mean(error[:, sl] ** 2)))
            for name, sl in (
                ("legs", slice(0, 12)),
                ("waist", slice(12, 15)),
                ("arms", slice(15, 29)),
            )
        },
        "joint_motion_active_count": int(active.sum()),
        "joint_motion_correlation": float(correlation[active].mean()) if active.any() else 0.0,
        "joint_motion_correlation_per_joint": correlation.tolist(),
        "joint_motion_definition": "Pearson correlation over executed reached states and matching reference frames; active reference joints have temporal std >= 0.1 rad. Constant output gives zero correlation.",
    }
