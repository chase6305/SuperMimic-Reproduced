"""NumPy motion references and orientation helpers without the policy runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.schema import open_motion


def rotation(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q_wxyz)[..., [1, 2, 3, 0]])


def gravity(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz)
    return rotation(q).inv().apply(np.broadcast_to([0.0, 0.0, -1.0], q.shape[:-1] + (3,)))


def heading_error(actual_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    """Signed actual-minus-reference yaw error, wrapped to [-pi, pi]."""

    def yaw(q):
        w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    delta = yaw(actual_wxyz) - yaw(reference_wxyz)
    return np.arctan2(np.sin(delta), np.cos(delta))


class MotionReference:
    def __init__(self, path: Path, default_q: np.ndarray):
        with open_motion(path) as (data, info):
            self.info = info
            self.body_names = data["body_names"].tolist()
            root = self.body_names.index("pelvis")
            self.q = data["joint_pos"].copy()
            self.qd = data["joint_vel"].copy()
            self.root_pos = data["body_pos_w"][:, root].copy()
            self.root_quat = data["body_quat_w"][:, root].copy()
            self.root_lin = data["body_lin_vel_w"][:, root].copy()
            self.root_ang = data["body_ang_vel_w"][:, root].copy()
            anchor = self.body_names.index("torso_link")
            self.anchor_pos = data["body_pos_w"][:, anchor].copy()
            self.anchor_quat = data["body_quat_w"][:, anchor].copy()
        self.frames = len(self.q)
        r = rotation(self.root_quat)
        # Same yaw extraction as yaw_quat: preserve heading, discard pitch/roll.
        w, x, y, z = self.root_quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        zeros = np.zeros_like(yaw)
        heading = Rotation.from_quat(
            np.stack([zeros, zeros, np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)
        )
        self.features = np.concatenate(
            [
                self.q - default_q,
                self.qd,
                gravity(self.root_quat),
                r.inv().apply(self.root_ang),
                heading.inv().apply(self.root_lin),
                self.root_pos[:, 2:3],
            ],
            axis=1,
        ).astype(np.float32)

    def goal(self, frame: int, future_frames: int) -> np.ndarray:
        indices = np.minimum(np.arange(future_frames) + frame, self.frames - 1)
        return self.features[indices].reshape(-1)
