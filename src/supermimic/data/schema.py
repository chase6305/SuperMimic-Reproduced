"""Validate the G1 motion ABI before allocating simulator or device tensors."""

from __future__ import annotations

import math
import os
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from supermimic.interfaces import G1_BODY_NAMES, G1_JOINT_NAMES
from supermimic.numpy_io import inspect_npz_headers

_ARRAY_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
)
_REQUIRED_FIELDS = frozenset(("fps", "joint_names", "body_names", *_ARRAY_FIELDS))


@dataclass(frozen=True)
class MotionPlan:
    """CPU metadata for one validated publication, without retaining its arrays.

    Reuse checks the opened inode, size and modification time. This is not a
    content snapshot: callers must keep the input immutable between phases.
    """

    path: Path
    info: Mapping[str, Any]
    array_dtypes: Mapping[str, np.dtype]
    identity: tuple[int, int, int, int]
    uncompressed_bytes: int


def _identity(stream) -> tuple[int, int, int, int]:
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _check_identity(stream, expected, path: Path) -> None:
    if _identity(stream) != expected:
        raise RuntimeError(f"Motion publication changed during reading: {path}")


def _inspect_headers(data, path: Path) -> dict[str, np.dtype]:
    """Check ABI dimensions before allocating arrays, including zero-width strings."""
    if missing := _REQUIRED_FIELDS - set(data.files):
        raise ValueError(f"Missing motion fields {sorted(missing)}: {path}")
    headers = inspect_npz_headers(data, _REQUIRED_FIELDS)
    if math.prod(headers["fps"].shape) != 1:
        raise ValueError(f"Expected one fps value: {path}")
    if headers["joint_names"].shape != (29,):
        raise ValueError(f"G1 joint order mismatch: {path}")
    if headers["body_names"].shape != (30,):
        raise ValueError(f"Expected the 30 unique G1 body names: {path}")
    shape = headers["joint_pos"].shape
    if len(shape) != 2 or shape[1] != 29 or shape[0] < 2:
        raise ValueError(f"Expected at least two frames of 29 joint positions: {path}")
    frames = shape[0]
    for key, expected in {
        "joint_pos": (frames, 29),
        "joint_vel": (frames, 29),
        "body_pos_w": (frames, 30, 3),
        "body_quat_w": (frames, 30, 4),
        "body_lin_vel_w": (frames, 30, 3),
        "body_ang_vel_w": (frames, 30, 3),
    }.items():
        if headers[key].shape != expected or headers[key].dtype.kind not in "fi":
            raise ValueError(f"Invalid {key}, expected finite numeric {expected}: {path}")
    return {key: headers[key].dtype for key in _ARRAY_FIELDS}


@contextmanager
def _opened_motion(path, *, expected_fps, max_uncompressed_bytes, plan=None):
    path = Path(path)
    if plan is not None and path.absolute() != plan.path:
        raise ValueError(f"Motion plan belongs to a different path: {path}")
    with path.open("rb") as stream:
        identity = _identity(stream)
        if plan is not None and identity != plan.identity:
            raise RuntimeError(f"Motion publication changed since inspection: {path}")
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            unpacked = sum(info.file_size for info in entries)
            if unpacked > max_uncompressed_bytes:
                raise ValueError(f"Motion exceeds per-clip unpacked byte budget: {path}")
            if len({entry.filename for entry in entries}) != len(entries):
                raise ValueError(f"Duplicate NPZ entries: {path}")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            dtypes = _inspect_headers(data, path)
            if plan is None:
                info = _validate_arrays(data, path, expected_fps=expected_fps)
            else:
                if abs(plan.info["fps"] - expected_fps) > 1e-4:
                    raise ValueError(f"Expected {expected_fps} Hz motion, got {plan.info['fps']}: {path}")
                info = {**plan.info, "body_names": list(plan.info["body_names"])}
            _check_identity(stream, identity, path)
            try:
                yield data, info, dtypes, identity, unpacked
            finally:
                _check_identity(stream, identity, path)


def inspect_motion(
    path: Path | str, *, expected_fps: float = 50.0, max_uncompressed_bytes: int = 512 * 1024**2
) -> MotionPlan:
    """Fully validate one publication and retain only its reusable CPU metadata."""
    # Keep `..` after symlink components: collapsing it lexically can select a
    # different file from the path supplied by the caller.
    path = Path(path).absolute()
    with _opened_motion(
        path, expected_fps=expected_fps, max_uncompressed_bytes=max_uncompressed_bytes
    ) as (_, info, dtypes, identity, unpacked):
        return MotionPlan(
            path=path,
            info=MappingProxyType({**info, "body_names": tuple(info["body_names"])}),
            array_dtypes=MappingProxyType(dtypes),
            identity=identity,
            uncompressed_bytes=unpacked,
        )


@contextmanager
def open_motion(
    path: Path | str, *, expected_fps: float = 50.0, max_uncompressed_bytes: int = 512 * 1024**2,
    plan: MotionPlan | None = None,
):
    """Yield ``(data, info)`` with ZIP admission and array reads on one opened inode.

    Atomic replacement of the pathname cannot redirect reads after admission.
    The caller must not modify the opened file's contents in place. Read archive
    members within this context; the NPZ and its descriptor close on exit. A
    plan skips repeated value validation only for the same opened publication.
    """
    with _opened_motion(
        path, expected_fps=expected_fps, max_uncompressed_bytes=max_uncompressed_bytes, plan=plan
    ) as (data, info, _, _, _):
        yield data, info


def validate_motion(
    path: Path | str, *, expected_fps: float = 50.0, max_uncompressed_bytes: int = 512 * 1024**2
) -> dict:
    with open_motion(
        path, expected_fps=expected_fps, max_uncompressed_bytes=max_uncompressed_bytes
    ) as (_, info):
        return info


def _validate_arrays(data, path: Path, *, expected_fps: float) -> dict:
    if missing := _REQUIRED_FIELDS - set(data.files):
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
    joint_pos = data["joint_pos"]
    shape = joint_pos.shape
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
        value = joint_pos if key == "joint_pos" else data[key]
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
