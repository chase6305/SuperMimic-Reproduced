"""Force sampling distinguishes flight intervals from mere clip completion."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic.deployment.metrics import airborne_metrics


def test_airborne_metrics_import_and_run_without_tracking_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = {'torch', 'mujoco', 'mujoco_warp', 'mjlab', 'warp'}
class BlockTracking(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or (
            fullname == 'supermimic.deployment.tracking_metrics'
        ):
            raise AssertionError('Unexpected tracking dependency: ' + fullname)
sys.meta_path.insert(0, BlockTracking())
import numpy as np
from supermimic.deployment.metrics import airborne_metrics
stats = airborne_metrics(np.zeros((8, 13)), 0.005)
assert stats['airborne_substeps'] == 8
assert stats['longest_airborne_s'] == 0.04
assert stats['airborne_intervals_ge_40ms'] == 1
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        check=True,
        timeout=20,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_tracking_metric_legacy_import_and_star_export_preserve_identity():
    from supermimic.deployment.metrics import tracking_metrics

    canonical = importlib.import_module("supermimic.deployment.tracking_metrics")
    assert tracking_metrics is canonical.tracking_metrics
    exported = {}
    exec("from supermimic.deployment.metrics import *", {}, exported)
    assert set(exported) == {"airborne_metrics", "tracking_metrics"}
    assert exported["tracking_metrics"] is canonical.tracking_metrics
    assert exported["airborne_metrics"] is airborne_metrics


def test_unknown_metric_attribute_is_rejected():
    metrics = importlib.import_module("supermimic.deployment.metrics")
    with pytest.raises(AttributeError, match="unknown_metric"):
        _ = metrics.unknown_metric


def test_airborne_runs_reject_contact_and_short_gaps():
    contacts = np.zeros((28, 13))
    contacts[:, 3] = 100  # left vertical force
    contacts[2:10, 3] = 0  # 40 ms of both feet unloaded
    contacts[12:15, 3] = 0  # only 15 ms
    contacts[18:28, 3] = 0
    contacts[18:28, 10] = 1  # hand/other ground support is not flight
    stats = airborne_metrics(contacts, 0.005)
    assert stats["airborne_substeps"] == 11
    assert stats["longest_airborne_s"] == 0.04
    assert stats["airborne_intervals_ge_40ms"] == 1
    contacts[:, 6] = 100  # right foot alone supports the body
    assert airborne_metrics(contacts, 0.005)["airborne_substeps"] == 0
    assert airborne_metrics(np.zeros((0, 13)), 0.005)["longest_airborne_s"] == 0


def test_airborne_intervals_and_force_peaks_across_contact_chunks(monkeypatch):
    from supermimic.deployment import metrics

    monkeypatch.setattr(metrics, "_CONTACT_CHUNK_SIZE", 8)
    contacts = np.zeros((29, 13), dtype=np.float64)
    contacts[:, 1] = 2.0
    contacts[5:14, 1] = 0.0
    contacts[21:27, 1] = 0.0
    contacts[19, 7:10] = (1.0, 4.0, 0.0)
    stats = airborne_metrics(contacts, 0.005)
    assert stats["airborne_substeps"] == 15
    assert stats["airborne_time_s"] == 0.005 * 15
    assert stats["longest_airborne_s"] == 0.005 * 9
    assert stats["airborne_intervals_ge_40ms"] == 1
    assert metrics._max_ground_force(contacts) == 5.0
    assert metrics._max_ground_force(contacts[:0]) == 0.0
