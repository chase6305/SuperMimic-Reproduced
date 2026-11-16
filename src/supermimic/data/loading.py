"""Plan validated NPZ clips, then fill CPU storage without retaining all clips."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supermimic.data.reference import REF_DIM, ClipInfo, reference_array
from supermimic.data.schema import MotionPlan, inspect_motion, open_motion

_VALIDATION_ROWS = 65_536
_BODY_FIELDS = {
    "pos": "body_pos_w",
    "quat": "body_quat_w",
    "lin": "body_lin_vel_w",
    "ang": "body_ang_vel_w",
}


@dataclass(frozen=True)
class NpzMotions:
    ref: np.ndarray
    bodies: dict[str, np.ndarray]
    clips: tuple[ClipInfo, ...]
    plans: tuple[MotionPlan, ...]


def load_npz_motions(
    files: Iterable[Path | str], body_names: Sequence[str], *, compact: bool = False
) -> NpzMotions:
    """Load each validated clip into preallocated arrays with legacy dtype rules.

    References are converted to float32 per clip before assignment. Body fields
    preserve NumPy's cross-clip concatenate promotion independently of each
    other. Inputs must remain immutable between inspection and consumption.
    """
    plans = tuple(inspect_motion(path) for path in files)
    if not plans:
        raise ValueError("Motion loading needs at least one clip")
    requested = tuple(body_names)
    # Resolve every clip's own ordering; duplicate requested bodies are valid.
    indices = [
        [plan.info["body_names"].index(name) for name in requested] for plan in plans
    ]
    frames = sum(plan.info["frames"] for plan in plans)
    ref = np.empty((frames, REF_DIM), dtype=np.float32)
    bodies = {}
    if not compact:
        for key, field in _BODY_FIELDS.items():
            dtype = np.result_type(*(plan.array_dtypes[field] for plan in plans))
            width = 4 if key == "quat" else 3
            bodies[key] = np.empty((frames, len(requested), width), dtype=dtype)
    clips = []
    start = 0
    for plan, body_idx in zip(plans, indices, strict=True):
        stop = start + plan.info["frames"]
        with open_motion(plan.path, plan=plan) as (data, info):
            arrays = {key: data[key] for key in plan.array_dtypes}
            with np.errstate(over="ignore", invalid="ignore"):
                chunk = reference_array(arrays, info["body_names"]).astype(np.float32)
            for begin in range(0, len(chunk), _VALIDATION_ROWS):
                if not np.isfinite(chunk[begin : begin + _VALIDATION_ROWS]).all():
                    raise ValueError(
                        f"Motion reference contains non-finite values after float32 conversion: {plan.path}"
                    )
            ref[start:stop] = chunk
            if not compact:
                for key, field in _BODY_FIELDS.items():
                    bodies[key][start:stop] = arrays[field][:, body_idx]
            del arrays, chunk
        clips.append(ClipInfo(plan.path.stem, start, stop - start))
        start = stop
    return NpzMotions(ref=ref, bodies=bodies, clips=tuple(clips), plans=plans)
