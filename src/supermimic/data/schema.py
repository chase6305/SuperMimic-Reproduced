"""Validate the G1 motion ABI before allocating simulator or device tensors."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np

from supermimic.interfaces import G1_BODY_NAMES, G1_JOINT_NAMES


def validate_motion(
    path: Path | str, *, expected_fps: float = 50.0, max_uncompressed_bytes: int = 512 * 1024**2
) -> dict:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if sum(info.file_size for info in entries) > max_uncompressed_bytes:
            raise ValueError(f"Motion exceeds per-clip unpacked byte budget: {path}")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError(f"Duplicate NPZ entries: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if missing := required - set(data.files):
            raise ValueError(f"Missing motion fields {sorted(missing)}: {path}")
        fps_array = data["fps"]
        if fps_array.size != 1:
            raise ValueError(f"Expected one fps value: {path}")
        fps = float(fps_array.reshape(-1)[0])
        if not math.isfinite(fps) or abs(fps - expected_fps) > 1e-4:
            raise ValueError(f"Expected {expected_fps} Hz motion, got {fps}: {path}")
        if tuple(data["joint_names"].tolist()) != G1_JOINT_NAMES:
            raise ValueError(f"G1 joint order mismatch: {path}")
        names = data["body_names"].tolist()
        if len(names) != 30 or set(names) != set(G1_BODY_NAMES):
            raise ValueError(f"Expected the 30 unique G1 body names: {path}")
        shape = data["joint_pos"].shape
        if len(shape) != 2 or shape[1] != 29 or shape[0] < 2:
            raise ValueError(f"Expected at least two frames of 29 joint positions: {path}")
        frames, bodies = shape[0], len(names)
        for key, expected in {
            "joint_pos": (frames, 29),
            "joint_vel": (frames, 29),
            "body_pos_w": (frames, bodies, 3),
            "body_quat_w": (frames, bodies, 4),
            "body_lin_vel_w": (frames, bodies, 3),
            "body_ang_vel_w": (frames, bodies, 3),
        }.items():
            value = data[key]
            if (
                value.shape != expected
                or value.dtype.kind not in "fi"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid {key}, expected finite numeric {expected}: {path}")
            if key == "body_quat_w" and not np.allclose(
                np.linalg.norm(value, axis=-1), 1, atol=2e-3
            ):
                raise ValueError(f"Non-unit wxyz body quaternions: {path}")
    return {
        "name": path.stem,
        "frames": frames,
        "fps": fps,
        "duration_s": (frames - 1) / fps,
        "body_names": names,
    }
