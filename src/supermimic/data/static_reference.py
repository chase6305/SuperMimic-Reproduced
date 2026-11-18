"""Construct an explicitly separate fixed-pose task from one validated motion frame."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np

from supermimic.artifacts import snapshot_file
from supermimic.data.schema import open_motion, validate_motion


def make_static_reference(source: Path, output: Path, *, frame: int, seconds: float) -> dict:
    """Freeze pose, zero velocities and retain source identity at the original 50 Hz.

    Repeated arrays are broadcast views; NumPy's ZIP writer streams them in
    bounded chunks. No duration-sized pose/velocity arrays are materialized.
    Publication refuses to replace an existing output, including a racing writer.
    """
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError("Static reference frame must be a nonnegative integer")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("Static reference seconds must be a finite number")
    if not math.isfinite(seconds) or not 30 <= seconds <= 1800:
        raise ValueError("Static reference seconds must be within [30, 1800]")
    controls = round(seconds * 50)
    if not math.isclose(seconds * 50, controls, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Static reference duration must be an exact number of 50 Hz controls")
    source, output = map(Path, (source, output))
    if output.suffix != ".npz":
        raise ValueError("Static reference output must have the .npz suffix")
    if output.exists():
        raise FileExistsError(output)
    source_info = validate_motion(source)
    if frame >= source_info["frames"]:
        raise ValueError("Static reference frame is outside the source motion")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".supermimic-static-", dir=output.parent) as directory:
        temporary = Path(directory)
        pinned = temporary / "input.npz"
        digest = snapshot_file(source, pinned)
        arrays = {}
        with open_motion(pinned) as (reference, info):
            if frame >= info["frames"]:
                raise ValueError("Static reference frame is outside the source motion")
            for name in ("joint_pos", "body_pos_w", "body_quat_w"):
                row = reference[name][frame].copy()
                arrays[name] = np.broadcast_to(row, (controls + 1, *row.shape))
            for name, pose in (
                ("joint_vel", "joint_pos"),
                ("body_lin_vel_w", "body_pos_w"),
                ("body_ang_vel_w", "body_pos_w"),
            ):
                row = np.zeros(arrays[pose].shape[1:], dtype=arrays[pose].dtype)
                arrays[name] = np.broadcast_to(row, (controls + 1, *row.shape))
            for name in ("joint_names", "body_names", "fps"):
                arrays[name] = reference[name].copy()
        arrays.update(
            source=np.array(
                "Separate fixed-pose task, not acceptance of the source dynamic motion"
            ),
            source_motion_sha256=np.array(digest),
            source_frame=np.array(frame, dtype=np.int64),
        )
        publication = temporary / "reference.npz"
        np.savez_compressed(publication, **arrays)
        info = validate_motion(publication)
        from supermimic.eval.static_hold import validate_static_reference

        with open_motion(publication) as (reference, _):
            validate_static_reference(reference)
        # link is atomic and exclusive on the same filesystem; temporary cleanup
        # removes only our staging name, leaving the successfully published inode.
        os.link(publication, output)
    return {"source_motion_sha256": digest, "source_frame": frame, **info, "name": output.stem}
