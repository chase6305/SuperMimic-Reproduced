"""Motion validation and array consumers share one admitted input inode."""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from supermimic.data import schema
from supermimic.data.schema import open_motion, validate_motion
from supermimic.interfaces import G1_BODY_NAMES, G1_JOINT_NAMES


def make_motion(path, *, frames=20, **overrides):
    quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
    quaternion[..., 0] = 1
    values = {
        "fps": np.array([50.0]),
        "joint_names": np.array(G1_JOINT_NAMES),
        "body_names": np.array(G1_BODY_NAMES),
        "joint_pos": np.zeros((frames, 29), dtype=np.float32),
        "joint_vel": np.zeros((frames, 29), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "body_quat_w": quaternion,
        "body_lin_vel_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 30, 3), dtype=np.float32),
    }
    np.savez(path, **{**values, **overrides})


def test_motion_schema_rejects_wrong_fps_joint_order_and_nan(tmp_path):
    path = tmp_path / "motion.npz"
    make_motion(path)
    assert validate_motion(path)["frames"] == 20
    with np.load(path) as values:
        valid = {key: values[key].copy() for key in values.files}
    for field, bad in [
        ("fps", [30.0]),
        ("joint_names", valid["joint_names"][::-1]),
        ("joint_pos", np.full((20, 29), np.nan)),
        ("body_quat_w", np.zeros((20, 30, 4))),
    ]:
        broken = tmp_path / (field + ".npz")
        np.savez(broken, **{**valid, field: bad})
        with pytest.raises(ValueError):
            validate_motion(broken)


def test_motion_unpack_budget_checked_before_arrays(tmp_path):
    path = tmp_path / "motion.npz"
    make_motion(path)
    with pytest.raises(ValueError, match="byte budget"):
        validate_motion(path, max_uncompressed_bytes=100)


@pytest.mark.parametrize("consumer", ["validate", "open"])
def test_atomic_replacement_keeps_admitted_archive_and_array_reads_together(
    tmp_path, monkeypatch, consumer
):
    path, replacement = tmp_path / "motion.npz", tmp_path / "replacement.npz"
    make_motion(path, frames=2)
    make_motion(replacement, frames=3)
    with zipfile.ZipFile(path) as archive:
        budget = sum(entry.file_size for entry in archive.infolist())
    with zipfile.ZipFile(replacement) as archive:
        assert sum(entry.file_size for entry in archive.infolist()) > budget
    original_load = np.load
    streams = []

    def rotate_before_array_load(source, **kwargs):
        replacement.replace(path)
        streams.append(source)
        return original_load(source, **kwargs)

    monkeypatch.setattr(schema.np, "load", rotate_before_array_load)
    if consumer == "validate":
        info = validate_motion(path, max_uncompressed_bytes=budget)
    else:
        with open_motion(path, max_uncompressed_bytes=budget) as (data, info):
            assert data["joint_pos"].shape == (2, 29)
            assert not streams[0].closed

    assert info["frames"] == 2
    assert streams[0].closed
    # The current pathname points at the replacement, which was not admitted.
    with original_load(path, allow_pickle=False) as data:
        assert data["joint_pos"].shape == (3, 29)
    with pytest.raises(ValueError, match="byte budget"):
        validate_motion(path, max_uncompressed_bytes=budget)


@pytest.mark.parametrize("outcome", ["success", "validation", "consumer", "budget", "archive"])
def test_motion_context_closes_descriptor_and_npz_on_every_exit(tmp_path, monkeypatch, outcome):
    path = tmp_path / "motion.npz"
    make_motion(path, frames=2, **({"fps": [30.0]} if outcome == "validation" else {}))
    if outcome == "archive":
        path.write_bytes(b"not a zip archive")
    streams, archives = [], []
    original_open, original_load = Path.open, np.load

    def record_open(open_path, *args, **kwargs):
        stream = original_open(open_path, *args, **kwargs)
        if open_path == path and args and args[0] == "rb":
            streams.append(stream)
        return stream

    def record_load(source, **kwargs):
        archive = original_load(source, **kwargs)
        archives.append(archive)
        return archive

    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(schema.np, "load", record_load)
    if outcome == "success":
        with open_motion(path) as (data, info):
            assert info["frames"] == 2 and data["joint_pos"].shape == (2, 29)
            assert len(streams) == 1 and not streams[0].closed
    elif outcome == "consumer":
        with pytest.raises(RuntimeError, match="consumer failed"):
            with open_motion(path):
                raise RuntimeError("consumer failed")
    elif outcome == "validation":
        with pytest.raises(ValueError, match="Expected 50.0 Hz"):
            with open_motion(path):
                raise AssertionError("invalid motion must not reach the consumer")
    elif outcome == "budget":
        with pytest.raises(ValueError, match="byte budget"):
            with open_motion(path, max_uncompressed_bytes=1):
                raise AssertionError("oversized motion must not reach the consumer")
    else:
        with pytest.raises(zipfile.BadZipFile):
            with open_motion(path):
                raise AssertionError("invalid archive must not reach the consumer")

    assert len(streams) == 1 and streams[0].closed
    assert all(archive.zip is None for archive in archives)
