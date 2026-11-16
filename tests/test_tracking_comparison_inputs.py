"""Comparison inputs keep the plotting ABI and bind arrays to their fingerprints."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic.artifacts import sha256
from supermimic.deployment import evidence, trajectory
from supermimic.deployment.evidence import read_motion_evidence

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_tracking.py"
_SPEC = importlib.util.spec_from_file_location("tracking_comparison", _SCRIPT)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


def _motion(path, *, offset=0):
    values = {
        "body_names": np.array(["pelvis"]),
        "body_pos_w": np.arange(12, dtype=np.float64).reshape(4, 1, 3) + offset,
        "body_quat_w": np.broadcast_to([1.0, 0, 0, 0], (4, 1, 4)).copy(),
        "joint_pos": np.arange(116, dtype=np.float64).reshape(4, 29) + offset,
    }
    np.savez(path, **values)
    return values


def _run(path, motion_hash, *, width=36, state_offset=0):
    path.mkdir()
    states = np.arange(3 * width, dtype=np.float64).reshape(3, width) + state_offset
    np.savez(
        path / "trajectory.npz", states=states,
        reference_frame=np.array([1, 2]), control_dt=np.array([0.02]),
    )
    result = {
        "status": "complete",
        "auto_reset": False,
        "motion_sha256": motion_hash,
        "trajectory_sha256": sha256(path / "trajectory.npz"),
        "reference_goal_offset": 1,
        "control_dt": 0.02,
        "completed_motion": False,
        "failure": "root_tilt",
    }
    (path / "result.json").write_text(json.dumps(result))
    return states, result


@pytest.mark.parametrize("width", [36, 71, 80])
def test_minimal_reference_and_unfinished_motion_need_no_actions_contacts_or_bundle(tmp_path, width):
    motion = tmp_path / "motion.npz"
    expected = _motion(motion)
    motion_hash, position, orientation, joints = comparison._comparison_reference(motion)
    states, expected_result = _run(tmp_path / "run", motion_hash, width=width)

    result, recorded = comparison._comparison_run(tmp_path / "run", motion_hash, len(joints))

    assert motion_hash == sha256(motion)
    np.testing.assert_array_equal(position, expected["body_pos_w"][:, 0])
    np.testing.assert_array_equal(orientation, expected["body_quat_w"][:, 0])
    np.testing.assert_array_equal(joints, expected["joint_pos"])
    assert result == expected_result and result["completed_motion"] is False
    assert recorded.sha256 == result["trajectory_sha256"]
    np.testing.assert_array_equal(recorded.states, states)
    assert set(recorded.arrays) == {"states", "reference_frame", "control_dt"}


def test_orientation_is_opt_in_and_default_motion_reader_still_accepts_three_fields(tmp_path):
    path = tmp_path / "three-fields.npz"
    values = _motion(path)
    del values["body_quat_w"]
    np.savez(path, **values)

    assert set(read_motion_evidence(path).arrays) == set(values)
    with pytest.raises(ValueError, match="body_quat_w"):
        comparison._comparison_reference(path)


def test_optional_orientation_shape_is_checked_before_any_array_allocation(tmp_path, monkeypatch):
    path = tmp_path / "motion.npz"
    values = _motion(path)
    values["body_quat_w"] = np.zeros((4, 1, 3))
    np.savez(path, **values)

    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed orientation reached array loading")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden)
    with pytest.raises(ValueError, match="body_quat_w"):
        comparison._comparison_reference(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("auto_reset", True),
        ("motion_sha256", "other-motion"),
        ("trajectory_sha256", "other-trajectory"),
        ("trajectory_sha256", None),
        ("control_dt", 0.04),
        ("reference_goal_offset", 0),
    ],
)
def test_known_identity_status_or_clock_mismatches_are_rejected(tmp_path, field, value):
    run = tmp_path / "run"
    _, result = _run(run, "motion")
    result[field] = value
    (run / "result.json").write_text(json.dumps(result))

    with pytest.raises(ValueError):
        comparison._comparison_run(run, "motion", 4)


@pytest.mark.parametrize("field", ["motion_sha256", "trajectory_sha256"])
def test_missing_fingerprints_do_not_downgrade_comparison_to_unverified(tmp_path, field):
    run = tmp_path / "run"
    _, result = _run(run, "motion")
    del result[field]
    (run / "result.json").write_text(json.dumps(result))

    with pytest.raises(ValueError, match="fingerprints"):
        comparison._comparison_run(run, "motion", 4)


def test_comparison_rejects_less_than_36_state_columns(tmp_path):
    run = tmp_path / "run"
    _run(run, "motion", width=35)
    with pytest.raises(ValueError, match="layout"):
        comparison._comparison_run(run, "motion", 4)


def test_comparison_rejects_missing_reached_state(tmp_path):
    run = tmp_path / "run"
    states, result = _run(run, "motion")
    np.savez(
        run / "trajectory.npz", states=states[:-1],
        reference_frame=np.array([1, 2]), control_dt=np.array(0.02),
    )
    result["trajectory_sha256"] = sha256(run / "trajectory.npz")
    (run / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="clock"):
        comparison._comparison_run(run, "motion", 4)


def test_motion_arrays_and_fingerprint_keep_opened_inode_after_path_rotation(tmp_path, monkeypatch):
    path, replacement = tmp_path / "motion.npz", tmp_path / "replacement.npz"
    expected = _motion(path)
    _motion(replacement, offset=100)
    digest, real_load = sha256(path), np.load

    def rotate(stream, **kwargs):
        replacement.replace(path)
        return real_load(stream, **kwargs)

    monkeypatch.setattr(evidence.np, "load", rotate)
    actual, position, orientation, joints = comparison._comparison_reference(path)

    assert actual == digest != sha256(path)
    np.testing.assert_array_equal(position, expected["body_pos_w"][:, 0])
    np.testing.assert_array_equal(orientation, expected["body_quat_w"][:, 0])
    np.testing.assert_array_equal(joints, expected["joint_pos"])


def test_run_arrays_and_fingerprint_keep_opened_inode_after_path_rotation(tmp_path, monkeypatch):
    run, other = tmp_path / "run", tmp_path / "other"
    states, expected_result = _run(run, "motion")
    _run(other, "motion", state_offset=100)
    real_load = np.load

    def rotate(stream, **kwargs):
        (other / "trajectory.npz").replace(run / "trajectory.npz")
        return real_load(stream, **kwargs)

    monkeypatch.setattr(trajectory.np, "load", rotate)
    result, recorded = comparison._comparison_run(run, "motion", 4)

    assert result == expected_result
    assert recorded.sha256 == result["trajectory_sha256"] != sha256(run / "trajectory.npz")
    np.testing.assert_array_equal(recorded.states, states)


def test_comparison_data_preparation_does_not_import_plotting_or_simulation_sdks(tmp_path):
    motion, run = tmp_path / "motion.npz", tmp_path / "run"
    _motion(motion)
    _run(run, sha256(motion))
    script = r"""
import importlib.abc
import importlib.util
import sys
from pathlib import Path

class BlockSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'mjlab', 'warp', 'scipy', 'matplotlib'}:
            raise AssertionError('Unexpected SDK import: ' + fullname)

sys.meta_path.insert(0, BlockSdk())
spec = importlib.util.spec_from_file_location('comparison', sys.argv[1])
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
digest, position, orientation, joints = comparison._comparison_reference(Path(sys.argv[2]))
result, recorded = comparison._comparison_run(Path(sys.argv[3]), digest, len(joints))
assert recorded.states.shape == (3, 36)
assert result['completed_motion'] is False
"""
    subprocess.run(
        [sys.executable, "-c", script, str(_SCRIPT), str(motion), str(run)],
        check=True, capture_output=True, text=True, timeout=30,
        env={
            **os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        },
    )
