"""Static pose acceptance uses physical states, complete duration and fixed limits."""

import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from supermimic.eval.static_hold import (
    judge_static_hold,
    static_hold_metrics,
    validate_static_reference,
)


def reference(frames=1501):
    result = {
        "joint_pos": np.zeros((frames, 29)),
        "joint_vel": np.zeros((frames, 29)),
        "body_pos_w": np.zeros((frames, 30, 3)),
        "body_quat_w": np.zeros((frames, 30, 4)),
        "body_lin_vel_w": np.zeros((frames, 30, 3)),
        "body_ang_vel_w": np.zeros((frames, 30, 3)),
    }
    result["body_pos_w"][:, 0, 2] = 1
    result["body_quat_w"][..., 0] = 1
    return result


def physical(frames=1501):
    states = np.zeros((frames, 71))
    states[:, 2] = states[:, 3] = 1
    return states, np.zeros(((frames - 1) * 4, 13))


def test_metrics_match_known_pose_offset_velocity_and_contact_counts():
    target = reference()
    states, contacts = physical()
    states[1:, :2] = [0.03, 0.04]
    states[1:, 2] += 0.02
    states[1:, 7:36] = 0.1
    states[1:, 42:] = 0.2
    states[1:, 3:7] = [math.cos(0.05), 0, 0, math.sin(0.05)]
    contacts[0, 10] = 1
    metrics = static_hold_metrics(states, target, 0, contacts)
    assert metrics == pytest.approx(
        {
            "root_xy_rmse_m": 0.05,
            "joint_rmse_rad": 0.1,
            "root_height_rmse_m": 0.02,
            "root_orientation_rmse_rad": 0.1,
            "nonfoot_ground_force_substeps": 1,
            "joint_velocity_rms_rad_s": 0.2,
        }
    )
    judgement = judge_static_hold(metrics, True, 30)
    assert not judgement["passed"]
    assert [c["name"] for c in judgement["checks"] if not c["passed"]] == [
        "nonfoot_ground_force_substeps"
    ]


@pytest.mark.parametrize(
    "field",
    ["joint_pos", "body_pos_w", "body_quat_w", "joint_vel", "body_lin_vel_w", "body_ang_vel_w"],
)
def test_moving_or_nonzero_velocity_reference_is_rejected(field):
    target = reference(3)
    target[field][1].flat[-1] = 0.01
    with pytest.raises(ValueError, match="Static hold requires"):
        validate_static_reference(target)


def test_quaternion_antipodes_are_static_but_zero_quaternions_are_invalid():
    target = reference(3)
    target["body_quat_w"][1] *= -1
    validate_static_reference(target)
    target["body_quat_w"][:] = 0
    with pytest.raises(ValueError, match="unit"):
        validate_static_reference(target)


@pytest.mark.parametrize("complete,seconds", [(True, 30), (False, 30), (True, 10)])
def test_full_at_least_thirty_second_hold_is_required(complete, seconds):
    target = reference()
    states, contacts = physical()
    metrics = static_hold_metrics(states, target, 0, contacts)
    judgement = judge_static_hold(metrics, complete, seconds)
    assert judgement["passed"] is (complete and seconds >= 30)
    assert judgement["criteria"]["version"] == "g1-static-pose-v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_xy_rmse_m", 0.101),
        ("joint_rmse_rad", 0.251),
        ("root_height_rmse_m", 0.051),
        ("root_orientation_rmse_rad", 0.151),
        ("nonfoot_ground_force_substeps", 1),
        ("joint_velocity_rms_rad_s", 0.501),
        ("joint_rmse_rad", float("nan")),
        ("joint_rmse_rad", float("inf")),
    ],
)
def test_every_pose_and_stability_condition_is_mandatory(field, value):
    states, contacts = physical()
    metrics = static_hold_metrics(states, reference(), 0, contacts)
    metrics[field] = value
    assert not judge_static_hold(metrics, True, 30)["passed"]


def test_missing_or_extra_measure_cannot_weaken_the_gate():
    states, contacts = physical()
    metrics = static_hold_metrics(states, reference(), 0, contacts)
    del metrics["joint_velocity_rms_rad_s"]
    with pytest.raises(ValueError, match="every fixed"):
        judge_static_hold(metrics, True, 30)
    metrics["unrecognized"] = 0
    with pytest.raises(ValueError, match="every fixed"):
        judge_static_hold(metrics, True, 30)


def load_assessor():
    path = Path(__file__).resolve().parents[1] / "scripts/assess_static_hold.py"
    spec = importlib.util.spec_from_file_location("static_hold_assessor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_scorer_recomputes_metrics_instead_of_trusting_result_json(monkeypatch):
    assessor = load_assessor()
    target = reference()
    states, contacts = physical()
    states[1:, 7:36] = 0.5

    @contextmanager
    def motion(path):
        assert path == Path("/pinned/reference.npz")
        yield target, {}

    monkeypatch.setattr("supermimic.data.schema.open_motion", motion)
    monkeypatch.setattr("supermimic.deployment.validation.validate_contract", lambda contract: None)
    inputs = SimpleNamespace(
        contract={},
        motion_path=Path("/pinned/reference.npz"),
        body_names=("pelvis",),
        trajectory=SimpleNamespace(
            states=states,
            arrays={"actions": np.zeros((1500, 29)), "physical_contacts": contacts},
            control_dt=0.02,
        ),
        result={"tracking": {"joint_rmse_rad": 0}},
        completed_motion=True,
    )
    metrics, judgement = assessor._score_inputs(inputs)
    assert metrics["joint_rmse_rad"] == 0.5
    assert not judgement["passed"]


def test_resource_admission_reserves_twenty_percent_before_reading_inputs(monkeypatch):
    assessor = load_assessor()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assess_static_hold.py",
            "--bundle",
            "/unused",
            "--motion",
            "/unused",
            "--run",
            "/unused",
            "--output",
            "/unused",
        ],
    )
    monkeypatch.setattr(
        "supermimic.resources.host_snapshot", lambda: {"ram_total_bytes": 100 * 1024**3}
    )

    class Admitted(Exception):
        pass

    def guard(output, *, limits):
        assert limits.reserve_ram_gib == 20 and limits.max_rss_gib == 2
        raise Admitted

    monkeypatch.setattr("supermimic.cpu_resources.cpu_guard", guard)
    with pytest.raises(Admitted):
        assessor.main()
