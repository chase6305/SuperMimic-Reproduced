"""Read selected trajectory arrays and their fingerprint from one opened NPZ.

This layer depends on NumPy, not policy or simulator SDKs. Tool policies choose
whether transitions, reference alignment or contact evidence are required.
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

from supermimic.numpy_io import ArrayHeader, inspect_npz_headers

PHYSICAL_CONTACT_COLUMNS = (
    "time_s",
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
    "nonfoot_fx",
    "nonfoot_fy",
    "nonfoot_fz",
    "nonfoot_force_contacts",
    "nonfoot_depth_m",
    "all_contacts",
)


@dataclass(frozen=True)
class Trajectory:
    arrays: Mapping[str, np.ndarray]
    sha256: str
    # None: contacts were not read. False: their column labels were absent.
    contact_layout_verified: bool | None = None
    control_dt_source: str = "trajectory"

    @property
    def states(self) -> np.ndarray:
        return self.arrays["states"]

    @property
    def control_dt(self) -> float:
        return float(self.arrays["control_dt"].item())


def _real_finite(value: np.ndarray) -> bool:
    return value.dtype.kind in "iuf" and bool(np.isfinite(value).all())


def _validate_headers(headers: Mapping[str, ArrayHeader]) -> None:
    states = headers["states"]
    if len(states.shape) != 2 or min(states.shape) < 1 or states.dtype.kind not in "iuf":
        raise ValueError("Trajectory states must be a numeric matrix with an initial row")
    if "control_dt" in headers:
        dt = headers["control_dt"]
        if math.prod(dt.shape) != 1 or dt.dtype.kind not in "iuf":
            raise ValueError("Trajectory control_dt must contain one numeric value")
    if "actions" in headers:
        actions = headers["actions"]
        if len(actions.shape) != 2 or actions.dtype.kind not in "iuf":
            raise ValueError("Trajectory actions must be a numeric matrix")
    if "reference_frame" in headers:
        frames = headers["reference_frame"]
        if len(frames.shape) != 1 or frames.dtype.kind not in "iu":
            raise ValueError("Trajectory reference_frame must be a one-dimensional integer array")
    if "physical_contacts" in headers:
        contacts = headers["physical_contacts"]
        if (
            len(contacts.shape) != 2
            or contacts.shape[1] != len(PHYSICAL_CONTACT_COLUMNS)
            or contacts.dtype.kind not in "iuf"
        ):
            raise ValueError(
                "Trajectory physical_contacts must be a numeric matrix with 13 columns"
            )
    for name, size in (("physical_contact_columns", 13), ("joint_names", 29)):
        if name in headers:
            value = headers[name]
            if value.shape != (size,) or value.dtype.kind not in "SU":
                raise ValueError(f"Trajectory {name} must contain {size} string labels")


def _validate_base(arrays: Mapping[str, np.ndarray]) -> bool | None:
    states, dt = arrays["states"], arrays["control_dt"]
    if states.ndim != 2 or min(states.shape) < 1 or not _real_finite(states):
        raise ValueError("Trajectory states must be a finite numeric matrix with an initial row")
    if dt.size != 1 or not _real_finite(dt) or float(dt.item()) <= 0:
        raise ValueError("Trajectory control_dt must contain one finite positive value")
    if "physical_contacts" not in arrays:
        return None
    contacts = arrays["physical_contacts"]
    if (
        contacts.ndim != 2
        or contacts.shape[1] != len(PHYSICAL_CONTACT_COLUMNS)
        or not _real_finite(contacts)
    ):
        raise ValueError(
            "Trajectory physical_contacts must be a finite numeric matrix with 13 columns"
        )
    columns = arrays.get("physical_contact_columns")
    if columns is None:
        return False
    if (
        columns.shape != (len(PHYSICAL_CONTACT_COLUMNS),)
        or tuple(columns.tolist()) != PHYSICAL_CONTACT_COLUMNS
    ):
        raise ValueError("Trajectory physical_contact_columns differ from the supported layout")
    return True


@contextmanager
def open_trajectory(
    path: Path | str,
    *,
    fields: Sequence[str] = ("states", "control_dt"),
    byte_budget: int | None = None,
    fallback_control_dt: float | None = None,
) -> Iterator[Trajectory]:
    """Load requested arrays, always including states and the control timestep.

    A budget bounds the selected ZIP members' total uncompressed bytes; unselected
    arrays are not loaded. All selected headers and known field shapes are checked
    before any array is allocated. Contact labels are also selected when contacts
    are requested and labels exist. No size restriction is imposed by default.

    Hashes before and after array loading identify the bytes actually consumed.
    The unbuffered descriptor remains open until context exit; arrays require no
    further file reads. Atomic pathname replacement is supported, while in-place
    mutation during reading is rejected. No trajectory-wide success or completion
    policy is inferred from shapes or contact counts.

    An explicit ``fallback_control_dt`` supports legacy assessment inputs that
    omitted that field. It is used only when the member is absent and is marked
    as ``control_dt_source='fallback'``; it never replaces a recorded value.
    """
    if isinstance(fields, str) or any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("Trajectory fields must be a sequence of nonempty names")
    if byte_budget is not None and (type(byte_budget) is not int or byte_budget < 0):
        raise ValueError("Trajectory byte budget must be a nonnegative integer")
    if fallback_control_dt is not None and (
        isinstance(fallback_control_dt, (bool, np.bool_))
        or not isinstance(fallback_control_dt, Real)
        or not math.isfinite(fallback_control_dt)
        or fallback_control_dt <= 0
    ):
        raise ValueError("Fallback control_dt must be a finite positive number")
    selected = tuple(dict.fromkeys(("states", "control_dt", *fields)))
    with Path(path).open("rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            by_name = {entry.filename: entry for entry in entries}
            if len(by_name) != len(entries):
                raise ValueError("Duplicate trajectory NPZ entries")
            # Match NpzFile's exact raw-name priority over the .npy alias.
            members = {name.removesuffix(".npy"): entry for name, entry in by_name.items()}
            members.update(by_name)
            if "physical_contacts" in selected and "physical_contact_columns" in members:
                selected = tuple(dict.fromkeys((*selected, "physical_contact_columns")))
            use_fallback = "control_dt" not in members and fallback_control_dt is not None
            if use_fallback:
                selected = tuple(name for name in selected if name != "control_dt")
            missing = [name for name in selected if name not in members]
            if missing:
                raise ValueError("Missing trajectory fields: " + ", ".join(missing))
            selected_members = {members[name].filename: members[name] for name in selected}
            unpacked = sum(entry.file_size for entry in selected_members.values())
            if byte_budget is not None and unpacked > byte_budget:
                raise ValueError("Trajectory selected arrays exceed the unpacked byte budget")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            _validate_headers(inspect_npz_headers(data, selected))
            arrays = {name: data[name] for name in selected}
            if use_fallback:
                arrays["control_dt"] = np.asarray(float(fallback_control_dt))
            if any(not isinstance(value, np.ndarray) for value in arrays.values()):
                raise ValueError("Requested trajectory fields must contain NumPy arrays")
            layout_verified = _validate_base(arrays)
            stream.seek(0)
            verified = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.fstat(stream.fileno())
            if (
                digest != verified
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise RuntimeError("Trajectory changed in place during reading")
            yield Trajectory(
                MappingProxyType(arrays), digest, layout_verified,
                "fallback" if use_fallback else "trajectory",
            )


def _required(record: Trajectory, name: str) -> np.ndarray:
    if name not in record.arrays:
        raise ValueError("Missing trajectory field: " + name)
    return record.arrays[name]


def validate_transitions(record: Trajectory, *, state_dim: int = 71, action_dim: int = 29) -> None:
    """Require initial/reached states aligned with complete control transitions.

    Zero controls with one initial state are valid. Physics contact rows may
    include a partial additional control and are deliberately not counted here.
    """
    states = record.states
    actions, frames = _required(record, "actions"), _required(record, "reference_frame")
    if states.ndim != 2 or states.shape[1] != state_dim:
        raise ValueError(f"Trajectory states must have {state_dim} columns")
    if actions.ndim != 2 or actions.shape[1] != action_dim or not _real_finite(actions):
        raise ValueError(
            f"Trajectory actions must be finite numeric rows with {action_dim} columns"
        )
    if frames.ndim != 1 or frames.dtype.kind not in "iu":
        raise ValueError("Trajectory reference_frame must be a one-dimensional integer array")
    if len(states) != len(actions) + 1 or len(frames) != len(actions):
        raise ValueError("Inconsistent trajectory transition lengths")


def validate_reference_clock(record: Trajectory, motion_frames: int, offset: int) -> None:
    """Require each reached state to use the advancing, clip-end-clamped target."""
    try:
        if isinstance(motion_frames, (bool, np.bool_)) or isinstance(offset, (bool, np.bool_)):
            raise TypeError
        motion_frames, offset = operator.index(motion_frames), operator.index(offset)
    except TypeError:
        raise ValueError("Reference motion length and offset must be integers") from None
    if motion_frames < 1 or offset < 0:
        raise ValueError("Reference motion length must be positive and offset nonnegative")
    frames = _required(record, "reference_frame")
    if frames.ndim != 1 or frames.dtype.kind not in "iu":
        raise ValueError("Trajectory reference_frame must be a one-dimensional integer array")
    last = motion_frames - 1
    start = min(offset, last)
    expected = start + np.minimum(np.arange(len(frames), dtype=np.int64), last - start)
    if not np.array_equal(frames, expected):
        raise ValueError("Reference clock discontinuity")
