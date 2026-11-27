"""NumPy contact metrics with lazy access to the full tracking scorer."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from supermimic.deployment.tracking_metrics import tracking_metrics as tracking_metrics

__all__ = ["airborne_metrics", "tracking_metrics"]

_CONTACT_CHUNK_SIZE = 65_536


def airborne_metrics(physical_contacts: np.ndarray, physics_dt: float) -> dict:
    """Observed unloaded intervals; these alone do not establish successful jumps."""
    if physics_dt <= 0 or not np.isfinite(physics_dt):
        raise ValueError("Physics timestep must be finite and positive")
    physical = np.asarray(physical_contacts)
    if physical.ndim != 2 or physical.shape[1] != 13:
        raise ValueError("Expected physical contact records with 13 columns")
    unloaded = np.empty(len(physical), dtype=bool)
    for begin in range(0, len(physical), _CONTACT_CHUNK_SIZE):
        end = begin + _CONTACT_CHUNK_SIZE
        rows = physical[begin:end]
        foot_load = np.linalg.norm(rows[:, 1:7].reshape(-1, 2, 3), axis=-1)
        unloaded[begin:end] = (foot_load < 1.0).all(-1) & (rows[:, 10] == 0)
    boundaries = np.flatnonzero(np.diff(unloaded, prepend=False, append=False))
    durations = (boundaries[1::2] - boundaries[::2]) * physics_dt
    samples = unloaded.sum()
    return {
        "airborne_substeps": int(samples),
        "airborne_time_s": float(samples * physics_dt),
        "longest_airborne_s": float(durations.max()) if len(durations) else 0.0,
        "airborne_intervals_ge_40ms": int(np.count_nonzero(durations >= 0.04)),
        "airborne_definition": "Each foot solved-force norm below 1 N, no nonfoot ground force; sampled every physics substep. Unloaded intervals are not a jump-success metric.",
    }


def _max_ground_force(physical: np.ndarray) -> float:
    peak = 0.0
    for begin in range(0, len(physical), _CONTACT_CHUNK_SIZE):
        rows = physical[begin : begin + _CONTACT_CHUNK_SIZE]
        total_force = rows[:, 1:10].reshape(-1, 3, 3).sum(axis=1)
        peak = np.maximum(peak, np.linalg.norm(total_force, axis=-1).max())
    return float(peak)


def __getattr__(name):
    if name == "tracking_metrics":
        return getattr(import_module("supermimic.deployment.tracking_metrics"), name)
    raise AttributeError(name)
