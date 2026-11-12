"""Pack planned NPZ clips into one CPU fp16 buffer with bounded pending work.

The input/RAM admission estimate preserves the command's existing policy. It is
not a peak-memory guarantee: workers also hold one clip's decoded arrays and
conversion temporaries, and publication needs serializer/validation buffers.
"""

from __future__ import annotations

import zipfile
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from supermimic.data.packed import PackedMotion, write_packed_motion
from supermimic.data.reference import REF_DIM, ClipInfo, reference_array
from supermimic.data.schema import MotionPlan, inspect_motion


def _admit(unpacked: int) -> None:
    from supermimic.resources import GIB, host_snapshot

    if unpacked > 2 * GIB or host_snapshot()["ram_available_bytes"] < 8 * GIB + unpacked * 3:
        raise RuntimeError("Packing would exceed the 2 GiB input or 8 GiB host RAM reserve")


def _load_reference(plan: MotionPlan) -> np.ndarray:
    from supermimic.data.schema import open_motion

    with open_motion(plan.path, plan=plan) as (data, info):
        # Packing has always converted directly to fp16. The float32
        # intermediate used by a training library would change rounding.
        return reference_array(data, info["body_names"]).astype(np.float16)


def pack_npz_motions(
    files: Iterable[Path | str], out: Path | str, *, workers: int = 2,
    progress: Callable[[int], object] | None = None,
) -> PackedMotion:
    """Preflight, fill and exclusively publish; report one increment per filled clip."""
    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four CPU packing workers")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    files = tuple(files)
    if not files:
        raise ValueError("No motion clips selected")
    # Preserve the command's cheap aggregate admission before any value scan.
    # Plans then measure the publications actually opened for validation, so a
    # pathname rotation cannot make the old central-directory estimate binding.
    unpacked = 0
    for path in files:
        with zipfile.ZipFile(path) as archive:
            unpacked += sum(entry.file_size for entry in archive.infolist())
    _admit(unpacked)
    plans = tuple(inspect_motion(path) for path in files)
    _admit(sum(plan.uncompressed_bytes for plan in plans))

    clips = []
    frames = 0
    for plan in plans:
        length = plan.info["frames"]
        clips.append(ClipInfo(plan.path.stem, frames, length))
        frames += length
    reference = np.empty((frames, REF_DIM), dtype=np.float16)
    remaining = iter(plans)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque(
            executor.submit(_load_reference, plan) for plan in islice(remaining, workers)
        )
        try:
            for clip in clips:
                future = pending.popleft()
                result = future.result()
                reference[clip.start : clip.end] = result
                # A completed Future also owns its result. Release both before
                # submitting the replacement so no extra clip remains live.
                del result, future
                if progress is not None:
                    progress(1)
                if (plan := next(remaining, None)) is not None:
                    pending.append(executor.submit(_load_reference, plan))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    packed = PackedMotion(ref=torch.from_numpy(reference), clips=tuple(clips))
    write_packed_motion(out, packed.ref, packed.clips)
    return packed
