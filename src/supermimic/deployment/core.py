"""CPU policy input and command construction; independent of the mjlab runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from supermimic.artifacts import sha256
from supermimic.deployment.reference import MotionReference as MotionReference
from supermimic.deployment.reference import gravity as gravity
from supermimic.deployment.reference import heading_error as heading_error
from supermimic.deployment.reference import rotation as rotation
from supermimic.deployment.validation import validate_contract
from supermimic.interfaces import ACT_DIM, OBS_DIM, OBS_OFFSETS


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
        if value.shape != (OBS_DIM,) or not np.isfinite(value).all():
            raise ValueError("Expected a finite 93-dimensional observation")
        if self.frames is None:
            self.frames = np.repeat(value[None], self.length, axis=0)
        else:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = value
        offsets = OBS_OFFSETS
        return np.concatenate(
            [
                self.frames[:, a:b].reshape(-1)
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ]
        )


class PolicyRunner:
    def __init__(self, bundle: Path):
        self.contract = json.loads((bundle / "contract.json").read_text())
        c = self.contract
        validate_contract(c)
        for name in ("policy.ts", "model.mjb"):
            if sha256(bundle / name) != c["files"][name]:
                raise ValueError(f"Bundle hash mismatch: {name}")
        torch.set_num_threads(1)
        self.actor = torch.jit.load(str(bundle / "policy.ts"), map_location="cpu").eval()
        self.history = ObservationHistory(c["history"])
        self.default_q = np.array(c["default_joint_pos"], dtype=np.float32)
        self.scale = np.array(c["action_scale"], dtype=np.float32)
        self.previous_action = np.zeros(ACT_DIM, dtype=np.float32)

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
        if action.shape != (ACT_DIM,) or not np.isfinite(action).all():
            raise ValueError("Policy returned invalid actions")
        self.previous_action[:] = action
        return action, self.default_q + self.scale * action
