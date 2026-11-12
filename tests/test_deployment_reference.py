"""Offline motion references do not import actor or simulator SDKs."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_reference_helpers_import_and_run_without_policy_or_simulator_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "mujoco", "mujoco_warp", "mjlab", "warp"}

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked or fullname == "supermimic.deployment.core":
            raise AssertionError(f"Offline reference imported runtime: {fullname}")

sys.meta_path.insert(0, NoRuntime())
from supermimic.deployment.reference import MotionReference, gravity, heading_error, rotation

identity = [1.0, 0.0, 0.0, 0.0]
assert gravity(identity).tolist() == [0.0, 0.0, -1.0]
assert heading_error(identity, identity) == 0.0
assert rotation(identity).as_quat().tolist() == [0.0, 0.0, 0.0, 1.0]
assert MotionReference.__module__ == "supermimic.deployment.reference"
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("name", ["MotionReference", "rotation", "gravity", "heading_error"])
def test_core_keeps_the_same_reference_classes_and_functions(name):
    reference = importlib.import_module("supermimic.deployment.reference")
    core = importlib.import_module("supermimic.deployment.core")
    assert getattr(core, name) is getattr(reference, name)
