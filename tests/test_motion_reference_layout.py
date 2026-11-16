"""Reference layout preserves motion fields without adopting physical-state frames."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic.data.reference import REF_DIM, REF_SLICES, REF_TERMS, ClipInfo, reference_array


def test_reference_layout_and_clip_metadata_import_without_numerical_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = {'numpy', 'torch', 'scipy', 'mujoco', 'mujoco_warp', 'mjlab', 'warp'}
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Reference metadata imported SDK: ' + fullname)
sys.meta_path.insert(0, BlockSDK())
from supermimic.data.reference import ClipInfo, REF_DIM, REF_SLICES, reference_array
assert REF_DIM == 71
assert REF_SLICES['root_ang_vel_w'] == slice(39, 42)
assert ClipInfo('walk', 10, 7).end == 17
assert callable(reference_array)
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        check=True,
        timeout=20,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_reference_slices_and_clip_extent_preserve_existing_layout():
    expected = {
        "root_pos_w": slice(0, 3),
        "root_quat_w": slice(3, 7),
        "joint_pos": slice(7, 36),
        "root_lin_vel_w": slice(36, 39),
        "root_ang_vel_w": slice(39, 42),
        "joint_vel": slice(42, 71),
    }
    assert REF_DIM == 71
    assert REF_SLICES == expected
    assert tuple(REF_SLICES) == tuple(name for name, _ in REF_TERMS)
    assert tuple(width for _, width in REF_TERMS) == (3, 4, 29, 3, 3, 29)
    clip = ClipInfo("walk", 11, 7)
    assert clip.end == 18
    clip.length = 9  # Existing ClipInfo is mutable.
    assert clip.end == 20


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_reference_pack_and_slices_preserve_nonzero_fields_and_world_angular_velocity(order):
    frames, bodies = 4, 3
    names = ("pelvis", "torso_link", "left_knee_link")
    data = {}
    for index, (name, width) in enumerate(
        (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3))
    ):
        data[name] = (
            np.arange(frames * bodies * width, dtype=np.float64).reshape(frames, bodies, width)
            + 100 * (index + 1)
            + 0.1234567890123
        )
    data["body_quat_w"] /= np.linalg.norm(data["body_quat_w"], axis=-1, keepdims=True)
    # Nonidentity rotation: body-frame angular velocity would differ from these
    # distinct world-frame components, which packing must retain unchanged.
    data["body_quat_w"][:, 0] = [0.5, 0.5, 0.5, 0.5]
    data["body_ang_vel_w"][:, 0] = [7.25, -3.5, 2.0]
    for index, name in enumerate(("joint_pos", "joint_vel")):
        data[name] = (
            np.arange(frames * 29, dtype=np.float64).reshape(frames, 29)
            + 10 * (index + 1)
            + 0.1234567890123
        )
    expected = {
        "root_pos_w": data["body_pos_w"][:, 0].copy(),
        "root_quat_w": data["body_quat_w"][:, 0].copy(),
        "joint_pos": data["joint_pos"].copy(),
        "root_lin_vel_w": data["body_lin_vel_w"][:, 0].copy(),
        "root_ang_vel_w": data["body_ang_vel_w"][:, 0].copy(),
        "joint_vel": data["joint_vel"].copy(),
    }
    reordered = {
        name: value[:, order, :].copy() if name.startswith("body_") else value.copy()
        for name, value in data.items()
    }
    packed = reference_array(reordered, [names[index] for index in order])
    assert packed.shape == (frames, 71)
    assert packed.dtype == np.float64  # Caller owns float32/float16 storage conversion.
    for name, value in expected.items():
        np.testing.assert_array_equal(packed[:, REF_SLICES[name]], value)
    restored = np.concatenate([packed[:, REF_SLICES[name]] for name, _ in REF_TERMS], axis=1)
    np.testing.assert_array_equal(restored, packed)
