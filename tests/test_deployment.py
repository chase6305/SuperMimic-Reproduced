"""Numerical parity of exported G1 controls and observation/goal layouts."""

import numpy as np
import torch

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    forward_kinematics,
    load_g1_model,
    resample_motion,
)
from supermimic.deployment.contract import build_model_and_contract
from supermimic.deployment.core import MotionReference, ObservationHistory, heading_error
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import flatten_history_term_major


def test_heading_error_wraps_and_ignores_quaternion_sign():
    def quat(degrees):
        angle = np.deg2rad(np.asarray(degrees)) / 2
        zeros = np.zeros_like(angle)
        return np.stack([np.cos(angle), zeros, zeros, np.sin(angle)], -1)

    actual, reference = quat([179, -179, 45]), quat([-179, 179, 15])
    expected = np.deg2rad([-2, 2, 30])
    np.testing.assert_allclose(heading_error(actual, reference), expected, atol=1e-12)
    np.testing.assert_allclose(heading_error(-actual, reference), expected, atol=1e-12)


def test_history_matches_training_including_reset():
    rng = np.random.default_rng(17)
    history = ObservationHistory(10)
    values = rng.normal(size=(20, 93)).astype(np.float32)
    expected = []
    for value in values:
        expected = (expected + [value])[-10:]
        backfilled = [expected[0]] * (10 - len(expected)) + expected
        flat = flatten_history_term_major(torch.tensor(np.array(backfilled))[:, None])
        np.testing.assert_array_equal(history.append(value), flat[0].numpy())
    history.reset()
    flat = history.append(values[0])
    expected_reset = flatten_history_term_major(
        torch.tensor(np.repeat(values[:1], 10, axis=0))[:, None]
    )
    np.testing.assert_array_equal(flat, expected_reset[0].numpy())


def make_motion(path):
    raw = np.zeros((20, 36))
    raw[:, 2] = 0.78
    raw[:, 6] = 1.0
    raw[:, 0] = np.arange(20) * 0.01
    raw[:, 7:] = np.linspace(0.0, 0.2, 20)[:, None]
    motion = resample_motion(raw, 50, 50)
    bodies = forward_kinematics(load_g1_model(), motion)
    np.savez(
        path,
        fps=[50.0],
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=bodies.body_names,
        joint_pos=motion.joint_pos,
        joint_vel=motion.joint_vel,
        body_pos_w=bodies.pos_w,
        body_quat_w=bodies.quat_w,
        body_lin_vel_w=bodies.lin_vel_w,
        body_ang_vel_w=bodies.ang_vel_w,
    )


def test_deployment_goals_match_training_and_clamp_at_clip_end(tmp_path):
    path = tmp_path / "motion.npz"
    make_motion(path)
    default_q = np.linspace(-0.2, 0.2, 29).astype(np.float32)
    reference = MotionReference(path, default_q)
    library = MotionLibrary([path], ("pelvis", "torso_link"), default_q, device="cpu")
    for frame in (0, 7, 17, 19):
        actual = reference.goal(frame, 10)
        expected = library.goal_window(torch.tensor([frame]), 10)[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_actuator_mapping_and_pd_parameters_match_training_scene():
    model, c = build_model_and_contract()
    assert c["joint_names"] == list(G1_JOINT_NAMES)
    assert sorted(c["actuator_joint_indices"]) == list(range(29))
    for actuator, joint in enumerate(c["actuator_joint_indices"]):
        assert (
            model.joint(int(model.actuator_trnid[actuator, 0])).name
            == "robot/" + c["joint_names"][joint]
        )
        assert c["kp"][joint] == model.actuator_gainprm[actuator, 0]
        assert c["kd"][joint] == -model.actuator_biasprm[actuator, 2]
        assert c["kp"][joint] > 0 and c["kd"][joint] > 0
    np.testing.assert_allclose(c["action_scale"], 0.25 * np.array(c["effort_limit"]) / c["kp"])


def test_motion_library_uses_each_clips_body_order(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    make_motion(first)
    with np.load(first) as source:
        arrays = {key: source[key].copy() for key in source.files}
    order = np.arange(30)[::-1]
    for key in ["body_names", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        arrays[key] = arrays[key][order] if key == "body_names" else arrays[key][:, order]
    np.savez(second, **arrays)
    lib = MotionLibrary([first, second], ("pelvis", "torso_link"), np.zeros(29), device="cpu")
    a, b = lib.bodies(torch.tensor([0])), lib.bodies(torch.tensor([20]))
    torch.testing.assert_close(a.pos_w, b.pos_w)
    torch.testing.assert_close(a.lin_vel_w, b.lin_vel_w)
    torch.testing.assert_close(lib.ref71(torch.tensor([0])), lib.ref71(torch.tensor([20])))
