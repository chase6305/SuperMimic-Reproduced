"""Preparation records the final copied motion while keeping preflight and output semantics."""

import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from supermimic.artifacts import sha256
from supermimic.data.preparation import prepare_motions
from supermimic.interfaces import G1_BODY_NAMES, G1_JOINT_NAMES


def _motion(path, *, frames=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    quaternions = np.zeros((frames, 30, 4), dtype=np.float32)
    quaternions[..., 0] = 1
    np.savez(
        path,
        fps=np.array([50.0]),
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(G1_BODY_NAMES),
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        body_pos_w=np.zeros((frames, 30, 3), dtype=np.float32),
        body_quat_w=quaternions,
        body_lin_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
    )


def test_prepare_writes_verified_manifest_and_keeps_copied_motion_writable(tmp_path):
    source = tmp_path / "source/clip.npz"
    _motion(source)
    source.chmod(0o444)
    output = tmp_path / "prepared"

    manifest = prepare_motions(source.parent, output, ["clip"])

    copied = output / source.name
    assert copied.read_bytes() == source.read_bytes()
    assert copied.stat().st_mode & stat.S_IWUSR
    assert source.stat().st_mode & 0o222 == 0
    assert manifest["robot"] == "unitree-g1-29dof"
    assert manifest["quaternion_order"] == "wxyz"
    assert manifest["files"][source.name] == {
        "name": "clip",
        "frames": 3,
        "fps": 50.0,
        "duration_s": 2 / 50,
        "body_names": list(G1_BODY_NAMES),
        "sha256": sha256(copied),
        "source": str(source.resolve()),
    }
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_prepare_metadata_matches_copy_if_source_is_replaced_after_preflight(
    tmp_path, monkeypatch
):
    from supermimic.data import schema

    source = tmp_path / "source/clip.npz"
    replacement = tmp_path / "replacement.npz"
    _motion(source, frames=3)
    _motion(replacement, frames=7)
    expected_digest = sha256(replacement)
    output = tmp_path / "prepared"
    original_validate = schema.validate_motion
    validated = []

    def replace_after_source_validation(path):
        info = original_validate(path)
        validated.append(path)
        if path == source:
            replacement.replace(source)
        return info

    monkeypatch.setattr(schema, "validate_motion", replace_after_source_validation)

    manifest = prepare_motions(source.parent, output, ["clip"])

    copied = output / source.name
    assert validated == [source, copied]
    assert sha256(copied) == expected_digest
    assert manifest["files"][source.name]["sha256"] == expected_digest
    assert manifest["files"][source.name]["frames"] == 7
    assert manifest["files"][source.name]["duration_s"] == 6 / 50
    assert manifest["files"][source.name]["source"] == str(source.resolve())
    assert original_validate(copied)["frames"] == 7
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_invalid_source_preflight_finishes_before_output_creation(tmp_path):
    source = tmp_path / "source/valid.npz"
    _motion(source)
    (source.parent / "invalid.npz").write_bytes(b"not a zip archive")
    output = tmp_path / "prepared"

    with pytest.raises(zipfile.BadZipFile):
        prepare_motions(source.parent, output, ["valid", "invalid"])

    assert not output.exists()


def test_prepare_rejects_duplicate_clip_names_before_creating_output(tmp_path):
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="unique"):
        prepare_motions(tmp_path / "source", output, ["clip", "clip"])
    assert not output.exists()


def test_prepare_never_overwrites_an_existing_output(tmp_path):
    source = tmp_path / "source/clip.npz"
    _motion(source)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("preserve existing data")

    with pytest.raises(FileExistsError):
        prepare_motions(source.parent, output, ["clip"])

    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve existing data"


def test_preparation_and_legacy_cli_entry_import_without_numerical_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = {"numpy", "scipy", "torch", "mujoco", "mujoco_warp", "mjlab", "warp"}

class NoSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError("Preparation imported a numerical SDK: " + fullname)

sys.meta_path.insert(0, NoSDKs())
from supermimic.data.preparation import prepare_motions
from supermimic.cli import prepare_motions as legacy_prepare_motions
assert prepare_motions is legacy_prepare_motions
assert "supermimic.data.schema" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
