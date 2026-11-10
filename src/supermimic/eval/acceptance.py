"""Project engineering gates for actual motion imitation (not paper benchmarks)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AcceptanceCfg:
    version: str = "g1-skills-v2-full-motion"
    min_full_clip_return: float = 0.7
    max_joint_rmse_rad: float = 0.25
    max_body_relative_pos_rmse_m: float = 0.15
    max_root_xy_rmse_m: float = 0.75
    max_root_orientation_rmse_rad: float = 0.3
    max_root_height_rmse_m: float = 0.08
    min_joint_motion_correlation: float = 0.6
    min_active_joints: int = 3
    min_reference_jump_events: int = 3
    min_jump_event_recall: float = 0.8
    min_jump_time_precision: float = 0.8
    min_jump_airtime_ratio: float = 0.6
    max_jump_airtime_ratio: float = 1.6
    max_jump_vertical_velocity_rmse_mps: float = 0.6
    min_joint_amplitude_ratio: float = 0.6
    max_joint_amplitude_ratio: float = 1.4
    max_worst_1s_joint_rmse_rad: float = 0.35
    max_worst_1s_root_orientation_rmse_rad: float = 0.4
    max_worst_1s_root_height_rmse_m: float = 0.1
    max_worst_1s_effector_rmse_m: float = 0.25
    min_inversion_recall: float = 0.8
    max_inverted_joint_rmse_rad: float = 0.3
    max_inverted_root_orientation_rmse_rad: float = 0.35
    max_inverted_root_height_rmse_m: float = 0.1


def judge_skill(
    metrics: dict, completed_motion: bool, skill: str, flight: dict | None = None
) -> dict:
    if skill not in ("jump", "dance", "fight"):
        raise ValueError("Expected jump, dance or fight")
    cfg = AcceptanceCfg()
    quality = metrics["quality"]
    checks = [
        {
            "name": "completed_motion",
            "value": bool(completed_motion),
            "passed": bool(completed_motion),
        }
    ]

    def check(name, value, limit, minimum=False):
        finite = bool(np.isfinite(value))
        checks.append(
            {
                "name": name,
                "value": float(value),
                "limit": limit,
                "comparison": ">=" if minimum else "<=",
                "passed": finite and bool(value >= limit if minimum else value <= limit),
            }
        )

    for name, limit in (
        ("joint_rmse_rad", cfg.max_joint_rmse_rad),
        ("body_relative_pos_rmse_m", cfg.max_body_relative_pos_rmse_m),
        ("root_xy_rmse_m", cfg.max_root_xy_rmse_m),
    ):
        check(name, metrics[name], limit)
    check(
        "full_clip_return",
        metrics["normalized_return_full_clip_zero_padded"],
        cfg.min_full_clip_return,
        True,
    )
    check(
        "root_orientation_rmse_rad",
        quality["root_orientation_rmse_rad"],
        cfg.max_root_orientation_rmse_rad,
    )
    check("root_height_rmse_m", quality["root_height_rmse_m"], cfg.max_root_height_rmse_m)
    check(
        "joint_motion_correlation",
        quality["joint_motion_correlation"],
        cfg.min_joint_motion_correlation,
        True,
    )
    check(
        "active_reference_joints", quality["joint_motion_active_count"], cfg.min_active_joints, True
    )
    for name, limit in (
        ("worst_1s_joint_rmse_rad", cfg.max_worst_1s_joint_rmse_rad),
        ("worst_1s_root_orientation_rmse_rad", cfg.max_worst_1s_root_orientation_rmse_rad),
        ("worst_1s_root_height_rmse_m", cfg.max_worst_1s_root_height_rmse_m),
    ):
        check(name, quality[name], limit)
    effectors = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    check(
        "worst_1s_effector_rmse_m",
        max(metrics["body_relative_worst_1s_rmse_m"][name] for name in effectors),
        cfg.max_worst_1s_effector_rmse_m,
    )
    for name, ratio in quality["joint_amplitude_ratio_by_group"].items():
        if ratio is not None:
            check(f"{name}_amplitude_min", ratio, cfg.min_joint_amplitude_ratio, True)
            check(f"{name}_amplitude_max", ratio, cfg.max_joint_amplitude_ratio)
    inversion = quality["inversion"]
    if inversion["reference_frames"]:
        check("inversion_recall", inversion["recall"], cfg.min_inversion_recall, True)
        check(
            "inverted_joint_rmse_rad", inversion["joint_rmse_rad"], cfg.max_inverted_joint_rmse_rad
        )
        check(
            "inverted_root_orientation_rmse_rad",
            inversion["root_orientation_rmse_rad"],
            cfg.max_inverted_root_orientation_rmse_rad,
        )
        check(
            "inverted_root_height_rmse_m",
            inversion["root_height_rmse_m"],
            cfg.max_inverted_root_height_rmse_m,
        )
    if skill == "jump":
        if flight is None:
            raise ValueError("Jump assessment requires reference/physical flight alignment")
        check(
            "reference_jump_events", flight["reference_events"], cfg.min_reference_jump_events, True
        )
        check("jump_event_recall", flight["event_recall"], cfg.min_jump_event_recall, True)
        check("jump_time_precision", flight["time_precision"], cfg.min_jump_time_precision, True)
        check("jump_airtime_ratio_min", flight["airtime_ratio"], cfg.min_jump_airtime_ratio, True)
        check("jump_airtime_ratio_max", flight["airtime_ratio"], cfg.max_jump_airtime_ratio)
        check(
            "root_vertical_velocity_rmse_mps",
            quality["root_vertical_velocity_rmse_mps"],
            cfg.max_jump_vertical_velocity_rmse_mps,
        )
    return {
        "passed": all(c["passed"] for c in checks),
        "skill": skill,
        "criteria": asdict(cfg),
        "scope": "Engineering imitation gate on this reference and initialization, not a general skill-success rate or paper benchmark",
        "checks": checks,
    }


def intervals(mask: np.ndarray, minimum_samples: int) -> list[tuple[int, int]]:
    changes = np.diff(
        np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(np.int8)
    )
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)
        if b - a >= minimum_samples
    ]


def flight_alignment(
    reference_clearance: np.ndarray, contacts: np.ndarray, control_dt: float, physics_dt: float
) -> dict:
    """Match significant target-flight intervals to actual unloaded physics substeps."""
    reference_events = intervals(
        (reference_clearance > 0.03).all(-1), max(1, int(np.ceil(0.12 / control_dt)))
    )
    force = np.linalg.norm(contacts[:, 1:7].reshape(-1, 2, 3), axis=-1)
    raw = (force < 1).all(-1) & (contacts[:, 10] == 0)
    actual_events = intervals(raw, max(1, int(np.ceil(0.04 / physics_dt))))
    actual = np.zeros(len(raw), dtype=bool)
    for a, b in actual_events:
        actual[a:b] = True
    window = np.zeros(len(raw), dtype=bool)
    hits = 0
    times = contacts[:, 0]
    expected_time = sum((b - a) * control_dt for a, b in reference_events)
    for a, b in reference_events:
        region = (times >= a * control_dt - 0.1) & (times <= b * control_dt + 0.1)
        window |= region
        hits += int(
            np.count_nonzero(actual & region) * physics_dt >= min(0.06, (b - a) * control_dt * 0.3)
        )
    actual_time = float(actual.sum() * physics_dt)
    return {
        "reference_events": len(reference_events),
        "matched_events": hits,
        "actual_events": len(actual_events),
        "event_recall": hits / max(len(reference_events), 1),
        "time_precision": float((actual & window).sum() / max(actual.sum(), 1)),
        "reference_airtime_s": expected_time,
        "actual_airtime_s": actual_time,
        "airtime_ratio": actual_time / max(expected_time, 1e-9),
        "definition": "Target: both feet collider bottoms >3 cm, >=120 ms. Actual: both feet <1 N and no nonfoot support, >=40 ms. Match tolerance +/-100 ms and >=min(60 ms,30% target duration) overlap.",
    }
