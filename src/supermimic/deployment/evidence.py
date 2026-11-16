"""Read diagnostic inputs and compare their recorded artifact fingerprints.

Matching a result proves agreement with that record, not independent provenance
or a successful rollout. The motion reader accepts the minimal fields needed by
offline path/joint plots; it does not validate the complete training motion ABI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from supermimic.numpy_io import inspect_npz_headers


def read_json_evidence(path: Path | str) -> tuple[dict, str]:
    """Decode and hash the same JSON bytes, including after pathname rotation."""
    with Path(path).open("rb", buffering=0) as stream:
        payload = stream.read()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def read_optional_result(trajectory: Path | str) -> tuple[dict | None, str | None]:
    try:
        return read_json_evidence(Path(trajectory).with_name("result.json"))
    except FileNotFoundError:
        return None, None


def verify_artifact_identity(result: dict | None, **actual: str) -> dict:
    """Reject known mismatches and identify absent legacy evidence explicitly."""
    if result is None:
        return {"status": "unverified", "reason": "result.json absent", "matched_fields": []}
    matched, missing = [], []
    for field, digest in actual.items():
        if field not in result:
            missing.append(field)
        elif result[field] != digest:
            raise ValueError(f"Artifact identity mismatch: {field}")
        else:
            matched.append(field)
    if missing:
        return {
            "status": "unverified",
            "reason": "result.json lacks fingerprints",
            "missing_fields": missing,
            "matched_fields": matched,
        }
    return {
        "status": "verified",
        "scope": "provided inputs match result.json fingerprints",
        "matched_fields": matched,
    }


def verify_control_dt(control_dt: float, **records: dict | None) -> None:
    """Compare any supplied timestep declaration without inventing missing ones."""
    for name, record in records.items():
        if record is None or "control_dt" not in record:
            continue
        value = record["control_dt"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not math.isclose(control_dt, value, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(f"Trajectory control_dt differs from {name}")


@dataclass(frozen=True)
class MotionEvidence:
    arrays: Mapping[str, np.ndarray]
    sha256: str


def read_motion_evidence(
    path: Path | str, *, include_orientation: bool = False
) -> MotionEvidence:
    """Read diagnostic motion fields and their same-inode SHA.

    A full motion may contain other fields, which are left unloaded. No new
    archive size ceiling is imposed on existing plotting inputs. Orientation
    comparisons explicitly request body_quat_w; other plots need only three
    fields and retain that smaller input contract.
    """
    fields = ("body_names", "body_pos_w", "joint_pos")
    numeric_fields = ("body_pos_w", "joint_pos")
    if include_orientation:
        fields += ("body_quat_w",)
        numeric_fields += ("body_quat_w",)
    with Path(path).open("rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        with zipfile.ZipFile(stream) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError("Duplicate diagnostic motion NPZ entries")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            headers = inspect_npz_headers(data, fields)
            names, positions, joints = (
                headers[field] for field in ("body_names", "body_pos_w", "joint_pos")
            )
            if (
                len(positions.shape) != 3
                or positions.shape[2] != 3
                or min(positions.shape) < 1
                or positions.dtype.kind not in "iuf"
            ):
                raise ValueError("Diagnostic body_pos_w must contain numeric frames/bodies/XYZ")
            if names.shape != (positions.shape[1],) or names.dtype.kind not in "SU":
                raise ValueError("Diagnostic body_names must match body_pos_w")
            if (
                len(joints.shape) != 2
                or joints.shape[0] != positions.shape[0]
                or joints.shape[1] < 1
                or joints.dtype.kind not in "iuf"
            ):
                raise ValueError("Diagnostic joint_pos must contain numeric motion frames")
            if include_orientation:
                quaternions = headers["body_quat_w"]
                if (
                    quaternions.shape != (*positions.shape[:2], 4)
                    or quaternions.dtype.kind not in "iuf"
                ):
                    raise ValueError("Diagnostic body_quat_w must match motion frames/bodies/WXYZ")
            arrays = {field: data[field] for field in fields}
            if any(not np.isfinite(arrays[field]).all() for field in numeric_fields):
                raise ValueError("Diagnostic motion arrays must be finite")
        stream.seek(0)
        verified = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        if (
            verified != digest
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError("Diagnostic motion changed in place during reading")
    return MotionEvidence(MappingProxyType(arrays), digest)
