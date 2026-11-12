"""Validated CPU compact-motion archives, independent of environment construction.

Legacy archives contain ``ref``, ``names``, ``starts`` and ``lengths``. New writes
also declare the motion conventions. Validation neither converts storage dtype
nor moves reference tensors onto a device. Unit-quaternion checks include the
storage dtype's quantization allowance and the norm computation's rounding.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import torch

from supermimic.data.reference import REF_DIM, REF_SLICES, REF_TERMS, ClipInfo
from supermimic.interfaces import QUATERNION_ORDER, ROBOT_ID
from supermimic.torch_io import read_torch_archive, write_torch_archive

_VALIDATION_ROWS = 65_536


@dataclass(frozen=True)
class PackedMotion:
    ref: torch.Tensor
    clips: tuple[ClipInfo, ...]


def _format_metadata() -> dict:
    return {
        "version": 1,
        "robot": ROBOT_ID,
        "quaternion_order": QUATERNION_ORDER,
        "fps": 50.0,
        "root_linear_velocity_frame": "world",
        "root_angular_velocity_frame": "world",
        "reference_terms": [list(term) for term in REF_TERMS],
    }


def _integer(value: Any, name: str) -> int:
    dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or dtype == torch.bool or getattr(dtype, "kind", None) == "b":
        raise ValueError(f"Packed motion {name} must be an integer, not a boolean")
    try:
        return operator.index(value)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"Packed motion {name} must be an integer") from error


def _validate_format(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Packed motion motion_format must be a dictionary")
    if _integer(value.get("version"), "format version") != 1:
        raise ValueError("Unsupported packed motion format version")
    for name in (
        "robot",
        "quaternion_order",
        "root_linear_velocity_frame",
        "root_angular_velocity_frame",
    ):
        actual = value.get(name)
        if not isinstance(actual, str) or actual != _format_metadata()[name]:
            raise ValueError(f"Unsupported packed motion format {name}")
    fps = value.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps != 50.0:
        raise ValueError("Unsupported packed motion format fps: expected 50 Hz")
    try:
        terms = tuple(tuple(term) for term in value["reference_terms"])
        matches = terms == REF_TERMS
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid packed motion reference terms") from error
    if not matches:
        raise ValueError("Unsupported packed motion reference terms")


def _table(value: Any, name: str) -> list:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"Packed motion {name} must be a table")
    try:
        return list(value)
    except TypeError as error:
        raise ValueError(f"Packed motion {name} must be a table") from error


def _validate_archive(data: Any) -> PackedMotion:
    if not isinstance(data, Mapping):
        raise ValueError("Expected a packed motion dictionary")
    if missing := {"ref", "names", "starts", "lengths"} - data.keys():
        raise ValueError(f"Missing packed motion fields: {sorted(missing)}")
    if "motion_format" in data:
        _validate_format(data["motion_format"])
    ref = data["ref"]
    if (
        not isinstance(ref, torch.Tensor)
        or ref.device.type != "cpu"
        or ref.layout != torch.strided
        or ref.is_nested
        or not ref.is_floating_point()
        or ref.ndim != 2
        or ref.shape[0] < 1
        or ref.shape[1] != REF_DIM
    ):
        raise ValueError(f"Packed motion ref must be a dense real floating CPU tensor (N>0, {REF_DIM})")
    names = _table(data["names"], "names")
    starts = _table(data["starts"], "starts")
    lengths = _table(data["lengths"], "lengths")
    if not names or len(names) != len(starts) or len(names) != len(lengths):
        raise ValueError("Packed motion clip tables must have equal nonzero lengths")
    clips = []
    end = 0
    for name, start, length in zip(names, starts, lengths, strict=True):
        if not isinstance(name, str):
            raise ValueError("Packed motion clip names must be strings")
        start, length = _integer(start, "clip start"), _integer(length, "clip length")
        if start != end or length < 1:
            raise ValueError("Packed motion clips must have positive lengths and contiguous starts")
        end = start + length
        clips.append(ClipInfo(name, start, length))
    if end != ref.shape[0]:
        raise ValueError("Packed motion clips must cover every reference frame exactly once")
    # Keep finite masks and quaternion arithmetic bounded independently of N.
    # Half/bfloat storage is checked in float32 to avoid rounding its norm to one.
    # NPZ accepts small norm errors before packing. Storage rounding can move a
    # valid boundary quaternion beyond that base tolerance without changing its
    # meaning, so account for representation and arithmetic errors separately.
    base_atol = 2e-3
    compute_dtype = torch.float32 if ref.element_size() < 4 else ref.dtype
    quat_atol = (
        base_atol
        + torch.finfo(ref.dtype).eps * (1 + base_atol)
        + 4 * torch.finfo(compute_dtype).eps
    )
    detached = ref.detach()
    for begin in range(0, ref.shape[0], _VALIDATION_ROWS):
        block = detached[begin : begin + _VALIDATION_ROWS]
        if block.element_size() < 4:
            block = block.float()
        if not torch.isfinite(block).all():
            raise ValueError("Packed motion ref contains non-finite values")
        norm = torch.linalg.vector_norm(block[:, REF_SLICES["root_quat_w"]], dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=quat_atol, rtol=1e-5):
            raise ValueError("Packed motion root quaternions must be unit wxyz quaternions")
    return PackedMotion(ref=ref, clips=tuple(clips))


def read_packed_motion(
    path: Path | str, *, mmap: bool | Literal["auto"] = "auto"
) -> PackedMotion:
    """Read trusted local archives on CPU and validate before device allocation."""
    return _validate_archive(read_torch_archive(path, mmap=mmap).data)


def write_packed_motion(
    path: Path | str, ref: torch.Tensor, clips: Iterable[ClipInfo]
) -> None:
    """Validate and exclusively publish a compact archive with explicit conventions."""
    clips = tuple(clips)
    data = {
        "ref": ref,
        "names": [clip.name for clip in clips],
        "starts": [clip.start for clip in clips],
        "lengths": [clip.length for clip in clips],
        "motion_format": _format_metadata(),
    }
    validated = _validate_archive(data)
    data["starts"] = [clip.start for clip in validated.clips]
    data["lengths"] = [clip.length for clip in validated.clips]
    write_torch_archive(path, data)
