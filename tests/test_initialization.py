"""Initial-state coordinates, reproducible local randomness and evidence replay."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from test_deployment import make_motion

from supermimic.artifacts import sha256
from supermimic.deployment.action_replay import verify_action_replay
from supermimic.deployment.export import export_policy
from supermimic.deployment.initialization import InitialPerturbation, parse_initialization_plan
from supermimic.deployment.mujoco import deploy
from supermimic.policy.nets import Policy, PolicyCfg


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"unknown": 1},
        {"root_translation_m": [0, 0]},
        {"root_translation_m": [0, 0, float("nan")]},
        {"root_translation_m": [True, 0, 0]},
        {"root_translation_m": [0, 0, 0.101]},
        {"joint_position_noise_std_rad": -0.01},
        {"joint_position_noise_std_rad": float("inf")},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.0},
    ],
)
def test_invalid_perturbations_are_rejected(value):
    with pytest.raises(ValueError):
        InitialPerturbation.from_mapping(value)


def test_world_rotation_is_left_multiplied_and_velocity_frames_are_explicit():
    from scipy.spatial.transform import Rotation

    # Start at 90 degrees yaw: a world X rotation differs from a body X rotation.
    data = SimpleNamespace(qpos=np.zeros(36), qvel=np.zeros(35))
    initial = Rotation.from_euler("z", np.pi / 2)
    data.qpos[3:7] = initial.as_quat()[[3, 0, 1, 2]]
    contract = {"root_qpos_index": 0, "root_qvel_index": 0}
    InitialPerturbation.from_mapping(
        {
            "root_translation_m": [0.01, -0.02, 0.005],
            "root_world_rotation_vector_rad": [0.03, 0, 0],
            "root_linear_velocity_delta_m_s": [0.1, -0.1, 0],
            "root_body_angular_velocity_delta_rad_s": [0, 0.1, -0.2],
        }
    ).apply(data, contract)
    expected = Rotation.from_rotvec([0.03, 0, 0]) * initial
    np.testing.assert_allclose(data.qpos[3:7], expected.as_quat()[[3, 0, 1, 2]])
    np.testing.assert_array_equal(data.qpos[:3], [0.01, -0.02, 0.005])
    np.testing.assert_array_equal(data.qvel[:6], [0.1, -0.1, 0, 0, 0.1, -0.2])


def test_joint_noise_does_not_change_global_rng_and_obeys_contract_soft_limits():
    contract = {
        "root_qpos_index": 0,
        "root_qvel_index": 0,
        "joint_qpos_indices": list(range(7, 36)),
        "joint_limits": [[-0.1, 0.1]] * 29,
        "initial_joint_soft_limit_factor": 0.5,
    }
    value = InitialPerturbation.from_mapping({"joint_position_noise_std_rad": 0.1, "seed": 21})
    first = SimpleNamespace(qpos=np.zeros(36), qvel=np.zeros(35))
    second = SimpleNamespace(qpos=np.zeros(36), qvel=np.zeros(35))
    np.random.seed(12)
    expected_global = np.random.random(3)
    np.random.seed(12)
    value.apply(first, contract)
    np.testing.assert_array_equal(np.random.random(3), expected_global)
    value.apply(second, contract)
    np.testing.assert_array_equal(first.qpos, second.qpos)
    expected = np.clip(np.random.Generator(np.random.PCG64(21)).normal(0, 0.1, 29), -0.05, 0.05)
    np.testing.assert_array_equal(first.qpos[7:], expected)


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [{"id": "../escape", "perturbation": {}}],
        [{"id": "baseline", "perturbation": {"root_translation_m": [0.01, 0, 0]}}],
        [{"id": "baseline", "perturbation": {}}, {"id": "baseline", "perturbation": {}}],
    ],
)
def test_plan_requires_zero_baseline_safe_unique_ids_and_nonempty_bounded_cases(cases):
    with pytest.raises(ValueError):
        parse_initialization_plan({"schema": "g1-static-initialization-v1", "cases": cases})


@pytest.fixture
def physical_case(tmp_path):
    torch.manual_seed(4)
    cfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "iteration": 0,
            "policy": Policy(cfg).state_dict(),
            "config": {"policy": asdict(cfg), "env": {"future_frames": 10, "anchor_obs": False}},
        },
        checkpoint,
    )
    bundle = tmp_path / "bundle"
    export_policy(checkpoint, bundle)
    motion = tmp_path / "motion.npz"
    make_motion(motion)
    run = tmp_path / "perturbed"
    value = InitialPerturbation.from_mapping(
        {"root_translation_m": [0.01, -0.01, 0], "joint_position_noise_std_rad": 0.01, "seed": 82}
    )
    result = deploy(bundle, motion, run, steps=4, initialization=value)
    return bundle, motion, run, result


def test_native_action_replay_matches_states_contacts_and_forces(physical_case):
    bundle, motion, run, result = physical_case
    assert result["initialization"]["seed"] == 82
    replay = verify_action_replay(bundle, motion, run, run / "replay.json")
    assert replay["passed"] and replay["controls"] == 4 and replay["physics_substeps"] == 16
    assert all(replay["fields_bitexact"].values())
    assert replay["declared_initialization_bitexact"]
    with np.load(run / "initialization.npz") as z:
        assert int(z["state_flags"]) == int(mujoco.mjtState.mjSTATE_INTEGRATION)


def test_zero_initialization_preserves_original_deployment_arrays(physical_case):
    bundle, motion, run, _ = physical_case
    default, zero = run.parent / "default", run.parent / "zero"
    deploy(bundle, motion, default, steps=4)
    deploy(bundle, motion, zero, steps=4, initialization=InitialPerturbation())
    with np.load(default / "trajectory.npz") as a, np.load(zero / "trajectory.npz") as b:
        assert a.files == b.files
        assert all(a[key].tobytes() == b[key].tobytes() for key in a.files)


def test_replay_rejects_modified_initial_snapshot(physical_case):
    bundle, motion, run, _ = physical_case
    with (run / "initialization.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_action_replay(bundle, motion, run, run / "replay.json")


def test_replay_detects_false_declaration_even_when_physics_replays_exactly(physical_case):
    bundle, motion, run, result = physical_case
    result["initialization"]["root_translation_m"] = [0, 0, 0]
    (run / "result.json").write_text(json.dumps(result))
    replay = verify_action_replay(bundle, motion, run, run / "replay.json")
    assert not replay["passed"] and not replay["declared_initialization_bitexact"]
    assert all(replay["fields_bitexact"].values())


def test_replay_detects_wrong_force_even_if_record_hash_is_updated(physical_case):
    bundle, motion, run, result = physical_case
    with np.load(run / "actuator-forces.npz") as z:
        forces = z["actuator_force"].copy()
    forces[0, 0] += 1
    np.savez(run / "actuator-forces.npz", actuator_force=forces)
    result["actuator_forces_sha256"] = sha256(run / "actuator-forces.npz")
    (run / "result.json").write_text(json.dumps(result))
    replay = verify_action_replay(bundle, motion, run, run / "replay.json")
    assert not replay["passed"]
    assert replay["fields_bitexact"] == {
        "states": True,
        "physical_contacts": True,
        "actuator_forces": False,
    }
    assert replay["first_mismatch"] == {"field": "actuator_forces", "index": 0}


def test_suite_preserves_quality_failures_and_continues_other_declared_cases(
    physical_case, monkeypatch
):
    from supermimic.eval.static_initialization import run_static_initialization_suite

    bundle, motion, run, _ = physical_case
    with np.load(motion) as z:
        arrays = {key: z[key] for key in z.files}
    for name in ("joint_pos", "body_pos_w", "body_quat_w"):
        arrays[name] = np.repeat(arrays[name][:1], 1501, axis=0)
    for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        arrays[name] = np.zeros((1501, *arrays[name].shape[1:]))
    static = run.parent / "static30.npz"
    np.savez(static, **arrays)
    plan = run.parent / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "g1-static-initialization-v1",
                "cases": [
                    {"id": "baseline", "perturbation": {}},
                    {"id": "shift", "perturbation": {"root_translation_m": [0.01, 0, 0]}},
                ],
            }
        )
    )

    # Four genuine controls per case deliberately fail the full 30-second gate.
    def incomplete(bundle, motion, output, *, initialization):
        return deploy(bundle, motion, output, steps=4, initialization=initialization)

    monkeypatch.setattr("supermimic.deployment.mujoco.deploy", incomplete)
    result = run_static_initialization_suite(bundle, static, plan, run.parent / "suite")
    assert result["status"] == "complete" and result["evaluated_cases"] == 2
    assert result["passes"] == 0 and result["all_replays_bitexact"]
    assert [row["id"] for row in result["records"]] == ["baseline", "shift"]
    for row in result["records"]:
        failed = {c["name"] for c in row["assessment"]["judgement"]["checks"] if not c["passed"]}
        assert {"completed_motion", "duration_s"} <= failed


def test_suite_rejects_dynamic_reference_before_output_or_policy_evaluation(physical_case):
    from supermimic.eval.static_initialization import run_static_initialization_suite

    bundle, motion, run, _ = physical_case
    plan = run.parent / "dynamic-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "g1-static-initialization-v1",
                "cases": [
                    {"id": "baseline", "perturbation": {}},
                ],
            }
        )
    )
    output = run.parent / "rejected"
    with pytest.raises(ValueError, match="constant"):
        run_static_initialization_suite(bundle, motion, plan, output)
    assert not output.exists()
