"""MotionLibrary keeps its values and queries after CPU loading is separated."""

import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from test_motion_loading import _motion, _old_concatenate_oracle

from supermimic.data.motion_library import MotionLibrary
from supermimic.data.reference import ClipInfo


def _clips(tmp_path):
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    _motion(paths[0], frames=3, dtypes={
        "body_pos_w": np.float32, "body_quat_w": np.float64,
        "body_lin_vel_w": np.int16, "body_ang_vel_w": np.float16,
    })
    _motion(paths[1], frames=4, order=range(29, -1, -1), offset=7, dtypes={
        "body_pos_w": np.float64, "body_quat_w": np.float32,
        "body_lin_vel_w": np.float64, "body_ang_vel_w": np.float32,
    })
    return paths


def _expected_goals(reference, q0):
    # The fixture's root q=(.5,.5,.5,.5) rotates x->y->z->x, with heading
    # yaw=pi/2. These explicit axes are independent of goal_features and the
    # quaternion helpers used by MotionLibrary.
    gravity = np.broadcast_to([0.0, -1.0, 0.0], (len(reference), 3))
    angular_body = reference[:, [40, 41, 39]]
    linear_heading = np.stack(
        [reference[:, 37], -reference[:, 36], reference[:, 38]], axis=-1
    )
    return np.concatenate([
        reference[:, 7:36] - q0, reference[:, 42:71], gravity, angular_body,
        linear_heading, reference[:, 2:3],
    ], axis=-1).astype(np.float32)


def _assert_queries(library, expected_ref, q0):
    assert library.clips == [ClipInfo("first", 0, 3), ClipInfo("second", 3, 4)]
    assert library.num_frames == 7 and library.num_clips == 2 and library.num_bins == 4
    for name, expected in {
        "clip_start": [0, 3], "clip_len": [3, 4], "clip_last": [2, 6],
        "frame_clip": [0, 0, 0, 1, 1, 1, 1], "clip_bin_offset": [0, 2],
        "bin_start": [0, 2, 3, 5], "bin_end": [2, 3, 5, 7], "bin_clip": [0, 0, 1, 1],
    }.items():
        assert getattr(library, name).tolist() == expected
    frames = torch.arange(7)
    assert library.bin_of_frame(frames).tolist() == [0, 0, 1, 2, 2, 3, 3]
    assert library.last_frame_of(frames).tolist() == [2, 2, 2, 6, 6, 6, 6]
    np.testing.assert_array_equal(library.ref71(frames).numpy(), expected_ref)

    goals = _expected_goals(expected_ref, q0)
    selected = [0, 2, 3, 6]
    indices = [[min(frame + step, 2 if frame < 3 else 6) for step in range(4)]
               for frame in selected]
    np.testing.assert_allclose(
        library.goal_window(torch.tensor(selected), 4).numpy(),
        goals[indices].reshape(4, 4 * 68), rtol=1e-6, atol=2e-5,
    )
    qpos, qvel = library.ref_qpos_qvel(frames)
    np.testing.assert_array_equal(qpos.numpy(), expected_ref[:, :36])
    expected_qvel = np.concatenate([
        expected_ref[:, 36:39], expected_ref[:, [40, 41, 39]], expected_ref[:, 42:71]
    ], axis=-1)
    np.testing.assert_allclose(qvel.numpy(), expected_qvel, rtol=1e-6, atol=2e-5)


def test_full_library_matches_old_concat_for_mixed_dtypes_and_reordered_clips(tmp_path):
    paths = _clips(tmp_path)
    requested = ("pelvis", "torso_link", "pelvis")
    expected_ref, expected_bodies = _old_concatenate_oracle(paths, requested)
    q0 = torch.linspace(-0.3, 0.3, 29)

    library = MotionLibrary(paths, requested, q0, device="cpu", compact=False, bin_frames=2)

    assert library._ref.dtype == torch.float32
    for field, key in {
        "body_pos_w": "pos", "body_quat_w": "quat",
        "body_lin_vel_w": "lin", "body_ang_vel_w": "ang",
    }.items():
        value = getattr(library, field)
        assert value.dtype == torch.float32
        np.testing.assert_array_equal(value.numpy(), expected_bodies[key].astype(np.float32))
    sampled = library.bodies(torch.tensor([6, 0, 3]))
    for field, key in {
        "pos_w": "pos", "quat_w": "quat", "lin_vel_w": "lin", "ang_vel_w": "ang"
    }.items():
        np.testing.assert_array_equal(
            getattr(sampled, field).numpy(), expected_bodies[key][[6, 0, 3]].astype(np.float32)
        )
    np.testing.assert_allclose(
        library.goal_feat.numpy(), _expected_goals(expected_ref, q0.numpy()), rtol=1e-6, atol=2e-5
    )
    _assert_queries(library, expected_ref, q0.numpy())


@pytest.mark.parametrize("store_dtype", [torch.float16, torch.float32])
def test_compact_library_keeps_float32_intermediate_and_goal_bin_queries(
    tmp_path, monkeypatch, store_dtype
):
    paths = _clips(tmp_path)
    requested = ("pelvis", "torso_link")
    old_ref, _ = _old_concatenate_oracle(paths, requested)
    q0 = torch.linspace(-0.3, 0.3, 29)
    model = object()
    constructed = []

    class QueryOnlyFK:
        def __init__(self, actual_model, *, device, dtype):
            assert actual_model is model and device == torch.device("cpu")
            assert dtype == torch.float32
            constructed.append(self)

        def body_indices(self, names):
            assert names == requested
            return torch.tensor([0, 1])

        def __call__(self, *args):
            raise AssertionError("Goal and bin queries must not invoke FK")

    converter = ModuleType("supermimic.data.motion_convert")
    converter.load_g1_model = lambda: model
    fk = ModuleType("supermimic.kinematics.g1_fk")
    fk.G1Kinematics = QueryOnlyFK
    monkeypatch.setitem(sys.modules, converter.__name__, converter)
    monkeypatch.setitem(sys.modules, fk.__name__, fk)

    library = MotionLibrary(
        paths, requested, q0, device="cpu", compact=True, store_dtype=store_dtype, bin_frames=2
    )

    assert len(constructed) == 1 and library._ref.dtype == store_dtype
    expected = old_ref.astype(np.float16 if store_dtype == torch.float16 else np.float32)
    np.testing.assert_array_equal(library._ref.numpy(), expected)
    if store_dtype == torch.float16:
        assert library._ref[0, 7].item() == 1.0
        assert np.float16(1 + 2**-11 + 2**-25) != 1.0
    _assert_queries(library, expected.astype(np.float32), q0.numpy())


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("tensor_q0", [False, True])
def test_bad_second_clip_is_rejected_before_first_device_call(
    tmp_path, monkeypatch, compact, tensor_q0
):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _motion(first)
    bad = _motion(second)
    bad["joint_vel"][-1, -1] = np.nan
    np.savez(second, **bad)
    q0 = torch.zeros(29) if tensor_q0 else np.zeros(29, dtype=np.float32)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Device conversion preceded admission of the second clip")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    monkeypatch.setattr(torch, "as_tensor", forbidden)
    with pytest.raises(ValueError, match="Invalid joint_vel"):
        MotionLibrary([first, second], ("pelvis",), q0, device="cuda", compact=compact)
    assert calls == []
