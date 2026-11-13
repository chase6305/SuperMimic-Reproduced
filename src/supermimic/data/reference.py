"""Flat motion-reference layout shared by NPZ readers and compact archives.

Both this reference record and the physical simulator state have 71 components.
They are different representations: ``root_ang_vel_w`` here is in the world
frame, while physical-state angular velocity is in the root body frame. Root
quaternions use wxyz. Packing does not rotate velocities or change array dtype.

The layout and clip metadata need only the standard library. NumPy is loaded
when an array is actually packed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

REF_TERMS = (
    ("root_pos_w", 3),
    ("root_quat_w", 4),
    ("joint_pos", 29),
    ("root_lin_vel_w", 3),
    ("root_ang_vel_w", 3),
    ("joint_vel", 29),
)
_REF_OFFSETS = tuple(accumulate((width for _, width in REF_TERMS), initial=0))
REF_DIM = _REF_OFFSETS[-1]
REF_SLICES = {
    name: slice(start, stop)
    for (name, _), start, stop in zip(
        REF_TERMS, _REF_OFFSETS[:-1], _REF_OFFSETS[1:], strict=True
    )
}


@dataclass
class ClipInfo:
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:  # exclusive
        return self.start + self.length


def reference_array(data: Mapping[str, Any], body_names: Sequence[str]) -> np.ndarray:
    """Pack validated motion arrays in their existing dtype and supplied body order.

    Callers keep ownership of validation and storage conversion: a full library
    uses float32, whereas a compact packer may store float16.
    """
    import numpy as np

    root = tuple(body_names).index("pelvis")
    return np.concatenate(
        [
            data["body_pos_w"][:, root],
            data["body_quat_w"][:, root],
            data["joint_pos"],
            data["body_lin_vel_w"][:, root],
            data["body_ang_vel_w"][:, root],
            data["joint_vel"],
        ],
        axis=1,
    )
