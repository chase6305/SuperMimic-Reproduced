"""A fixed engineering gate for one static reference pose, separate from dynamic skills."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StaticHoldCfg:
    version: str = "g1-static-pose-v1"
    min_duration_s: float = 30.0
    max_root_xy_rmse_m: float = 0.1
    max_joint_rmse_rad: float = 0.25
    max_root_height_rmse_m: float = 0.05
    max_root_orientation_rmse_rad: float = 0.15
    max_nonfoot_ground_force_substeps: int = 0
    max_joint_velocity_rms_rad_s: float = 0.5


def validate_static_reference(reference) -> None:
    """Require a validated motion ABI whose poses are fixed and velocities zero.

    Quaternion sign changes describe the same orientation. A microunit tolerance
    permits serialization roundoff, rather than accepting a moving reference.
    """
    tolerance = 1e-6
    for name in ("joint_pos", "body_pos_w"):
        values = np.asarray(reference[name], dtype=np.float64)
        if not np.isfinite(values).all() or np.max(np.abs(values - values[0])) > tolerance:
            raise ValueError(f"Static hold requires constant {name}")
    quaternions = np.asarray(reference["body_quat_w"], dtype=np.float64)
    if (
        not np.isfinite(quaternions).all()
        or np.max(np.abs(np.linalg.norm(quaternions, axis=-1) - 1)) > 1e-3
    ):
        raise ValueError("Static hold requires unit body_quat_w")
    distance = np.minimum(
        np.abs(quaternions - quaternions[0]).max(-1),
        np.abs(quaternions + quaternions[0]).max(-1),
    )
    if np.max(distance) > tolerance:
        raise ValueError("Static hold requires constant body_quat_w")
    for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        values = np.asarray(reference[name], dtype=np.float64)
        if not np.isfinite(values).all() or np.max(np.abs(values)) > tolerance:
            raise ValueError(f"Static hold requires zero {name}")


def static_hold_metrics(states, reference, root_index: int, contacts) -> dict:
    """Measure reached G1 states and solved contacts, never supplied result metrics.

    Input identity, state/action/force alignment and the reference clock are
    admitted by open_assessment_inputs before this numerical layer is called.
    """
    actual = np.asarray(states, dtype=np.float64)[1:]
    if actual.ndim != 2 or actual.shape[1] != 71 or not len(actual):
        raise ValueError("Static hold needs reached 71-dimensional G1 states")
    if not np.isfinite(actual).all():
        raise ValueError("Static hold needs finite states")
    positions = np.asarray(reference["body_pos_w"][0], dtype=np.float64)[root_index]
    joints = np.asarray(reference["joint_pos"][0], dtype=np.float64)
    target = np.asarray(reference["body_quat_w"][0], dtype=np.float64)[root_index]
    target_norm = np.linalg.norm(target)
    orientation = actual[:, 3:7]
    norms = np.linalg.norm(orientation, axis=-1, keepdims=True)
    if target_norm <= 1e-12 or np.any(norms <= 1e-12):
        raise ValueError("Static hold needs nonzero root quaternions")
    target = target / target_norm
    orientation = orientation / norms
    angle = 2 * np.arccos(np.clip(np.abs((orientation * target).sum(-1)), 0, 1))
    return {
        "root_xy_rmse_m": float(np.sqrt(np.mean(np.square(actual[:, :2] - positions[:2]).sum(-1)))),
        "joint_rmse_rad": float(np.sqrt(np.mean(np.square(actual[:, 7:36] - joints)))),
        "root_height_rmse_m": float(np.sqrt(np.mean(np.square(actual[:, 2] - positions[2])))),
        "root_orientation_rmse_rad": float(np.sqrt(np.mean(np.square(angle)))),
        "nonfoot_ground_force_substeps": int(np.count_nonzero(np.asarray(contacts)[:, 10] > 0)),
        "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(np.square(actual[:, 42:])))),
    }


def judge_static_hold(metrics: dict, completed_motion: bool, simulated_seconds: float) -> dict:
    cfg = StaticHoldCfg()
    required = {name.removeprefix("max_") for name in asdict(cfg) if name.startswith("max_")}
    if set(metrics) != required:
        raise ValueError("Static hold metrics must contain every fixed pose and stability measure")
    checks = [
        {
            "name": "completed_motion",
            "value": bool(completed_motion),
            "passed": bool(completed_motion),
        }
    ]
    checks.append(
        {
            "name": "duration_s",
            "value": float(simulated_seconds),
            "limit": cfg.min_duration_s,
            "comparison": ">=",
            "passed": bool(
                np.isfinite(simulated_seconds) and simulated_seconds >= cfg.min_duration_s
            ),
        }
    )
    for name, value in metrics.items():
        limit = getattr(cfg, "max_" + name)
        checks.append(
            {
                "name": name,
                "value": value,
                "limit": limit,
                "comparison": "<=",
                "passed": bool(np.isfinite(value) and value <= limit),
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "skill": "static-pose-hold",
        "criteria": asdict(cfg),
        "checks": checks,
        "scope": "Fixed reference pose for the recorded full duration and initialization. Separate from dynamic imitation, generalization, commanded locomotion and hardware validation.",
    }
