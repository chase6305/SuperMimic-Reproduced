"""G1 policy input layout shared by configuration, training and CPU deployment.

This module describes dimensions and ordering only. Tensor and NumPy numerical
implementations stay separate so that parity tests can compare them independently.
Importing this module must not initialize a simulator or a tensor runtime.
"""

from __future__ import annotations

from itertools import accumulate

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_BODY_NAMES = ("pelvis",) + tuple(
    "torso_link" if name == "waist_pitch_joint" else name.removesuffix("_joint") + "_link"
    for name in G1_JOINT_NAMES
)

ACT_DIM = len(G1_JOINT_NAMES)
CONTROL_DT = 0.02
OBS_TERMS = (
    ("joint_pos", ACT_DIM),
    ("joint_vel", ACT_DIM),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("actions", ACT_DIM),
)
OBS_DIM = sum(width for _, width in OBS_TERMS)
OBS_OFFSETS = tuple(accumulate((width for _, width in OBS_TERMS), initial=0))
GOAL_TERMS = (
    ("joint_pos", ACT_DIM),
    ("joint_vel", ACT_DIM),
    ("projected_gravity", 3),
    ("base_ang_vel", 3),
    ("heading_linear_velocity", 3),
    ("root_height", 1),
)
GOAL_FEAT_DIM = sum(width for _, width in GOAL_TERMS)
ANCHOR_OBS_DIM = 9
GOAL_DIM = 10 * GOAL_FEAT_DIM


def observation_dim(history: int) -> int:
    """Flattened term-major history width."""
    return OBS_DIM * history


def goal_dim(future_frames: int, anchor_obs: bool = False) -> int:
    """Future reference window followed by optional anchor-relative pose."""
    return GOAL_FEAT_DIM * future_frames + (ANCHOR_OBS_DIM if anchor_obs else 0)
