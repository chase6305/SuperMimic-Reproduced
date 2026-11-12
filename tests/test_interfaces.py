"""The policy interface remains usable before importing numerical SDKs."""

import os
import subprocess
import sys
from pathlib import Path

from supermimic.interfaces import (
    ACT_DIM,
    G1_BODY_NAMES,
    G1_JOINT_NAMES,
    GOAL_TERMS,
    OBS_OFFSETS,
    OBS_TERMS,
    goal_dim,
    observation_dim,
)


def test_interface_dimensions_and_robot_order():
    assert ACT_DIM == len(G1_JOINT_NAMES) == 29
    assert len(set(G1_BODY_NAMES)) == 30
    assert OBS_OFFSETS == (0, 29, 58, 61, 64, 93)
    assert tuple(name for name, _ in OBS_TERMS) == (
        "joint_pos",
        "joint_vel",
        "base_ang_vel",
        "projected_gravity",
        "actions",
    )
    assert tuple(width for _, width in GOAL_TERMS) == (29, 29, 3, 3, 3, 1)
    assert observation_dim(3) == 279
    assert goal_dim(4) == 272
    assert goal_dim(4, anchor_obs=True) == 281


def test_interface_and_contract_validation_do_not_import_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'mujoco', 'mjlab', 'scipy', 'warp'}:
            raise AssertionError('Heavy SDK imported: ' + fullname)
sys.meta_path.insert(0, BlockSDK())
from supermimic.interfaces import goal_dim
from supermimic.deployment.validation import validate_contract
assert goal_dim(2, True) == 145
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        check=True,
        timeout=20,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
