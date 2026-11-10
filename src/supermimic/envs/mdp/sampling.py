"""Optional reset priorities; retain the baseline distribution over the full clip."""

from __future__ import annotations

import math


def validate_priorities(windows, ratio):
    if not math.isfinite(ratio) or not 0 <= ratio <= 0.5:
        raise ValueError("Priority reset ratio must be within [0, 0.5]")
    result = tuple(tuple(window) for window in windows)
    for window in result:
        if (
            len(window) != 2
            or not all(math.isfinite(value) for value in window)
            or not 0 <= window[0] < window[1]
        ):
            raise ValueError("Priority windows require finite seconds with 0 <= start < end")
    if ratio and not result:
        raise ValueError("Positive reset priority requires at least one time window")
    return result


def priority_distribution(library, windows, ratio):
    windows = validate_priorities(windows, ratio)
    if not ratio:
        return None
    if library.num_clips != 1:
        raise ValueError("Time-window reset priority requires exactly one motion clip")
    import torch

    # Prepared motion libraries use the same fixed 50 Hz control clock.
    duration = (library.num_frames - 1) / 50
    mask = torch.zeros_like(library.bin_start, dtype=torch.bool)
    for start, end in windows:
        if end > duration + 1e-9:
            raise ValueError("Priority window extends beyond the original motion duration")
        mask |= (library.bin_start < end * 50) & (library.bin_end > start * 50)
    if not mask.any():
        raise ValueError("Priority windows do not intersect motion reset bins")
    weights = mask.float()
    return weights / weights.sum()
