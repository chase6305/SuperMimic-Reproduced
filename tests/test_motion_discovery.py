"""Motion selection stays lightweight while old library import paths remain valid."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from supermimic.data.discovery import discover_npz


def test_selection_filters_sorted_top_level_npz_paths_before_limiting(tmp_path):
    for name in ("gamma.npz", "alpha.npz", "beta.npz", "ignored.txt", "ignored.NPZ"):
        (tmp_path / name).touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.npz").touch()

    assert discover_npz(tmp_path) == [tmp_path / f"{name}.npz" for name in ("alpha", "beta", "gamma")]
    assert discover_npz(str(tmp_path), names=["gamma", "alpha", "alpha"], limit=1) == [
        tmp_path / "alpha.npz"
    ]
    assert discover_npz(tmp_path, names=[]) == []
    assert discover_npz(tmp_path, limit=0) == []


@pytest.mark.parametrize(
    ("manifest_text", "names", "expected"),
    [
        ("name,keep\ngamma,true\nbeta,0\nalpha,1\n", None, ["alpha", "gamma"]),
        ("name,keep\ngamma,True\nbeta,false\nalpha,1\n", ["gamma"], ["gamma"]),
        ("name\nbeta\nalpha\n", None, ["alpha", "beta"]),
    ],
)
def test_manifest_and_requested_names_keep_existing_selection_semantics(
    tmp_path, manifest_text, names, expected
):
    for name in ("alpha", "beta", "gamma"):
        (tmp_path / f"{name}.npz").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(manifest_text)

    assert discover_npz(tmp_path, names=names, manifest=manifest) == [
        tmp_path / f"{name}.npz" for name in expected
    ]


def test_discovery_import_and_execution_do_not_load_tensor_or_simulator_sdks(tmp_path):
    (tmp_path / "clip.npz").touch()
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "numpy", "mujoco", "mujoco_warp", "mjlab", "warp", "scipy"}

class NoSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked or fullname.startswith("supermimic.envs"):
            raise AssertionError(f"Discovery imported a runtime: {fullname}")

sys.meta_path.insert(0, NoSDKs())
from supermimic.data.discovery import discover_npz

motions = Path(sys.argv[2])
assert discover_npz(motions) == [motions / "clip.npz"]
assert "supermimic.data.motion_library" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "name",
    ["MotionLibrary", "ClipInfo", "RefState", "RefBodies", "goal_features", "REF_DIM", "GOAL_FEAT_DIM"],
)
def test_legacy_library_exports_are_the_same_data_layer_objects(name):
    canonical = importlib.import_module("supermimic.data.motion_library")
    legacy = importlib.import_module("supermimic.envs.mdp.motion_library")
    assert getattr(legacy, name) is getattr(canonical, name)


def test_library_discovery_exports_share_the_lightweight_implementation():
    canonical = importlib.import_module("supermimic.data.motion_library")
    legacy = importlib.import_module("supermimic.envs.mdp.motion_library")
    assert canonical.discover_npz is legacy.discover_npz is discover_npz
