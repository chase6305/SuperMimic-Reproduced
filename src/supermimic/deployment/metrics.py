"""NumPy contact metrics with lazy access to the full tracking scorer."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from supermimic.deployment.tracking_metrics import tracking_metrics as tracking_metrics

__all__ = ["airborne_metrics", "tracking_metrics"]


def airborne_metrics(physical_contacts: np.ndarray, physics_dt: float) -> dict:
    """Observed unloaded intervals; these alone do not establish successful jumps."""
    if physics_dt <= 0 or not np.isfinite(physics_dt):
        raise ValueError("Physics timestep must be finite and positive")
    physical = np.asarray(physical_contacts)
    if physical.ndim != 2 or physical.shape[1] != 13:
        raise ValueError("Expected physical contact records with 13 columns")
    foot_load = np.linalg.norm(physical[:, 1:7].reshape(-1, 2, 3), axis=-1)
    unloaded = (foot_load < 1.0).all(-1) & (physical[:, 10] == 0)
    changes = np.diff(np.concatenate([[False], unloaded, [False]]).astype(np.int8))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    durations = lengths * physics_dt
    return {
        "airborne_substeps": int(unloaded.sum()),
        "airborne_time_s": float(unloaded.sum() * physics_dt),
        "longest_airborne_s": float(durations.max()) if len(durations) else 0.0,
        "airborne_intervals_ge_40ms": int(np.count_nonzero(durations >= 0.04)),
        "airborne_definition": "Each foot solved-force norm below 1 N, no nonfoot ground force; sampled every physics substep. Unloaded intervals are not a jump-success metric.",
    }


def __getattr__(name):
    if name == "tracking_metrics":
        return getattr(import_module("supermimic.deployment.tracking_metrics"), name)
    raise AttributeError(name)
