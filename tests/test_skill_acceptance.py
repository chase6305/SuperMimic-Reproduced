"""Staying upright or taking off at the wrong time cannot pass the skill gate."""

import numpy as np

from supermimic.eval.acceptance import flight_alignment, judge_skill


def good_metrics():
    return {
        "joint_rmse_rad": 0.1,
        "body_relative_pos_rmse_m": 0.05,
        "root_xy_rmse_m": 0.2,
        "normalized_return_full_clip_zero_padded": 0.8,
        "body_relative_worst_1s_rmse_m": {
            name: 0.1
            for name in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
        },
        "quality": {
            "root_orientation_rmse_rad": 0.1,
            "root_height_rmse_m": 0.02,
            "joint_motion_correlation": 0.9,
            "joint_motion_active_count": 20,
            "root_vertical_velocity_rmse_mps": 0.1,
            "worst_1s_joint_rmse_rad": 0.1,
            "worst_1s_root_orientation_rmse_rad": 0.1,
            "worst_1s_root_height_rmse_m": 0.02,
            "joint_amplitude_ratio_by_group": {"legs": 1.0, "waist": None, "arms": 1.0},
            "inversion": {"reference_frames": 0},
        },
    }


def test_skill_gate_requires_full_duration_and_pose_dynamics():
    metrics = good_metrics()
    assert judge_skill(metrics, True, "dance")["passed"]
    assert not judge_skill(metrics, False, "dance")["passed"]
    metrics["quality"]["joint_motion_correlation"] = 0
    assert not judge_skill(metrics, True, "fight")["passed"]
    metrics["quality"]["joint_motion_correlation"] = 0.9
    metrics["joint_rmse_rad"] = 0.5
    assert not judge_skill(metrics, True, "dance")["passed"]


def test_jump_gate_matches_repeated_physical_flight_to_target_times():
    clearance = np.zeros((201, 2))
    contacts = np.zeros((800, 13))
    contacts[:, 0] = np.arange(1, 801) * 0.005
    contacts[:, 3] = 100
    for a, b in [(25, 40), (75, 90), (125, 140)]:
        clearance[a:b] = 0.06
        contacts[(contacts[:, 0] >= a * 0.02) & (contacts[:, 0] < b * 0.02), 3] = 0
    aligned = flight_alignment(clearance, contacts, 0.02, 0.005)
    assert aligned["reference_events"] == aligned["matched_events"] == 3
    assert aligned["time_precision"] == 1 and aligned["event_recall"] == 1
    assert judge_skill(good_metrics(), True, "jump", aligned)["passed"]
    contacts[:, 3] = 100
    grounded = flight_alignment(clearance, contacts, 0.02, 0.005)
    assert not judge_skill(good_metrics(), True, "jump", grounded)["passed"]
    for a, b in [(25, 40), (75, 90), (125, 140)]:
        contacts[(contacts[:, 0] >= a * 0.02 + 0.5) & (contacts[:, 0] < b * 0.02 + 0.5), 3] = 0
    shifted = flight_alignment(clearance, contacts, 0.02, 0.005)
    assert shifted["event_recall"] == 0 and shifted["time_precision"] == 0


def test_full_motion_gate_rejects_hidden_inversion_and_missing_kick_amplitude():
    metrics = good_metrics()
    metrics["quality"]["inversion"] = {
        "reference_frames": 53,
        "recall": 0.0,
        "joint_rmse_rad": 0.1,
        "root_orientation_rmse_rad": 1.2,
        "root_height_rmse_m": 0.02,
    }
    assert not judge_skill(metrics, True, "dance")["passed"]
    metrics = good_metrics()
    metrics["quality"]["joint_amplitude_ratio_by_group"]["legs"] = 0.05
    assert not judge_skill(metrics, True, "fight")["passed"]
    metrics = good_metrics()
    metrics["body_relative_worst_1s_rmse_m"]["left_ankle_roll_link"] = 0.8
    assert not judge_skill(metrics, True, "fight")["passed"]
