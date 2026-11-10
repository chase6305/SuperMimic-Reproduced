"""CPU policy input and command construction; independent of the mjlab runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from supermimic.artifacts import sha256
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.data.schema import validate_motion


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


class ObservationHistory:
    def __init__(self, length: int):
        if length < 1:
            raise ValueError("History length must be positive")
        self.length = length
        self.frames = None

    def reset(self) -> None:
        self.frames = None

    def append(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (93,) or not np.isfinite(value).all():
            raise ValueError("Expected a finite 93-dimensional observation")
        if self.frames is None:
            self.frames = np.repeat(value[None], self.length, axis=0)
        else:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = value
        offsets = (0, 29, 58, 61, 64, 93)
        return np.concatenate(
            [
                self.frames[:, a:b].reshape(-1)
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ]
        )


class MotionReference:
    def __init__(self, path: Path, default_q: np.ndarray):
        self.info = validate_motion(path)
        with np.load(path, allow_pickle=False) as data:
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


class PolicyRunner:
    def __init__(self, bundle: Path):
        self.contract = json.loads((bundle / "contract.json").read_text())
        c = self.contract
        if c.get("schema_version") != 1 or c.get("robot") != "unitree-g1-29dof":
            raise ValueError("Unsupported policy contract")
        if c["joint_names"] != list(G1_JOINT_NAMES) or c["history_layout"] != "term_major":
            raise ValueError("Unsupported joint order/history layout")
        for name in ("policy.ts", "model.mjb"):
            if sha256(bundle / name) != c["files"][name]:
                raise ValueError(f"Bundle hash mismatch: {name}")
        torch.set_num_threads(1)
        self.actor = torch.jit.load(str(bundle / "policy.ts"), map_location="cpu").eval()
        self.history = ObservationHistory(c["history"])
        self.default_q = np.array(c["default_joint_pos"], dtype=np.float32)
        self.scale = np.array(c["action_scale"], dtype=np.float32)
        self.previous_action = np.zeros(29, dtype=np.float32)

    def reset(self) -> None:
        self.history.reset()
        self.previous_action.fill(0)

    @torch.inference_mode()
    def act(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        root_quat: np.ndarray,
        angular_velocity_body: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.concatenate(
            [
                q - self.default_q,
                qd,
                angular_velocity_body,
                gravity(root_quat),
                self.previous_action,
            ]
        )
        history = self.history.append(observation)
        goal = np.asarray(goal, dtype=np.float32)
        if goal.shape != (self.contract["goal_dim"],) or not np.isfinite(goal).all():
            raise ValueError("Invalid policy goal")
        action = (
            self.actor(torch.from_numpy(history)[None], torch.from_numpy(goal)[None])[0]
            .numpy()
            .copy()
        )
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Policy returned invalid actions")
        self.previous_action[:] = action
        return action, self.default_q + self.scale * action
