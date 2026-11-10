"""A motion-faithfulness gate must reject a static pose on a changing target."""

import numpy as np

from supermimic.eval.quality import motion_quality, worst_window_rmse


def test_motion_quality_accepts_tracking_and_rejects_static_substitute():
    n = 200
    time = np.arange(n) / 50
    target = np.sin(time[:, None] * np.arange(1, 30)[None] * 0.5) * 0.4
    positions = np.zeros((n, 1, 3))
    positions[:, 0, 2] = 0.78
    orientations = np.zeros((n, 1, 4))
    orientations[:, :, 0] = 1
    reference = {
        "joint_pos": target,
        "body_pos_w": positions,
        "body_quat_w": orientations,
        "body_lin_vel_w": np.zeros((n, 1, 3)),
    }
    states = np.zeros((n + 1, 71))
    states[:, 2] = 0.78
    states[:, 3] = 1
    states[1:, 7:36] = target
    frames = np.arange(n)
    result = motion_quality(states, frames, reference, 0)
    assert result["joint_motion_correlation"] > 0.999999
    assert max(result["joint_group_rmse_rad"].values()) == 0
    assert all(
        abs(ratio - 1) < 1e-9
        for ratio in result["joint_amplitude_ratio_by_group"].values()
        if ratio is not None
    )
    assert result["root_height_rmse_m"] == result["root_orientation_rmse_rad"] == 0
    states[:, 7:36] = 0
    assert motion_quality(states, frames, reference, 0)["joint_motion_correlation"] == 0
    states[:, 3:7] *= -1
    assert motion_quality(states, frames, reference, 0)["root_orientation_rmse_rad"] == 0


def test_rolling_error_and_inversion_metrics_expose_rare_missed_stunts():
    squared = np.zeros(1000)
    squared[500:550] = 1
    assert np.sqrt(squared.mean()) < 0.3 and worst_window_rmse(squared, 50) == 1
    n = 7000
    states = np.zeros((n + 1, 71))
    states[:, 2] = 0.78
    states[:, 3] = 1
    quat = np.zeros((n, 1, 4))
    quat[:, :, 0] = 1
    quat[500:550, 0] = [0, 1, 0, 0]
    positions = np.zeros((n, 1, 3))
    positions[:, :, 2] = 0.78
    reference = {
        "joint_pos": np.zeros((n, 29)),
        "body_pos_w": positions,
        "body_quat_w": quat,
        "body_lin_vel_w": np.zeros((n, 1, 3)),
    }
    quality = motion_quality(states, np.arange(n), reference, 0)
    assert quality["root_orientation_rmse_rad"] < 0.3
    assert quality["inversion"]["reference_frames"] == 50 and quality["inversion"]["recall"] == 0
    assert quality["inversion"]["root_orientation_rmse_rad"] > 3
