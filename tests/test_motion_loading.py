"""NPZ planning bounds allocation and preserves the previous concatenate results."""

import gc
import io
import os
import weakref
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from supermimic.data import loading, schema
from supermimic.data.loading import load_npz_motions
from supermimic.data.reference import ClipInfo
from supermimic.data.schema import inspect_motion, open_motion, validate_motion
from supermimic.interfaces import G1_BODY_NAMES, G1_JOINT_NAMES

_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
)


def _motion(path, *, frames=2, order=None, dtypes=None, offset=0):
    order = tuple(range(30)) if order is None else tuple(order)
    dtypes = dtypes or {}
    values = {
        "fps": np.array([50.0]),
        "joint_names": np.array(G1_JOINT_NAMES),
        "body_names": np.array(G1_BODY_NAMES)[list(order)],
    }
    for key in _FIELDS:
        width = 4 if key == "body_quat_w" else 3 if key.startswith("body_") else 29
        shape = (frames, 30, width) if key.startswith("body_") else (frames, width)
        value = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 16 + offset
        if key == "body_quat_w":
            value[...] = [1, 0, 0, 0]
            value[:, 0] = [0.5, 0.5, 0.5, 0.5]
        if key == "joint_pos":
            # Direct float64->fp16 differs from the old float32 intermediate.
            value[0, 0] = 1 + 2**-11 + 2**-25
        if key.startswith("body_"):
            value = value[:, order]
        values[key] = value.astype(dtypes.get(key, np.float64))
    np.savez(path, **values)
    return values


def _old_concatenate_oracle(paths, body_names):
    references, bodies = [], {key: [] for key in ("pos", "quat", "lin", "ang")}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            names = data["body_names"].tolist()
            root = names.index("pelvis")
            index = np.array([names.index(name) for name in body_names])
            bp, bq, bl, ba = (data[key] for key in _FIELDS[2:])
            references.append(np.concatenate([
                bp[:, root], bq[:, root], data["joint_pos"], bl[:, root], ba[:, root],
                data["joint_vel"],
            ], axis=1).astype(np.float32))
            for key, array in zip(bodies, (bp, bq, bl, ba), strict=True):
                bodies[key].append(array[:, index])
    return np.concatenate(references, axis=0), {
        key: np.concatenate(chunks, axis=0) for key, chunks in bodies.items()
    }


@pytest.mark.parametrize("compact", [False, True])
def test_preallocated_loading_matches_old_concat_with_orders_dtypes_and_precision(tmp_path, compact):
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    _motion(paths[0], dtypes={
        "body_pos_w": np.float32, "body_quat_w": np.float64,
        "body_lin_vel_w": np.int16, "body_ang_vel_w": np.float16,
    })
    _motion(paths[1], frames=3, order=range(29, -1, -1), offset=7, dtypes={
        "body_pos_w": np.float64, "body_quat_w": np.float32,
        "body_lin_vel_w": np.float64, "body_ang_vel_w": np.float32,
    })
    requested = ("pelvis", "torso_link", "pelvis")
    expected_ref, expected_bodies = _old_concatenate_oracle(paths, requested)

    result = load_npz_motions(iter(paths), requested, compact=compact)

    assert result.ref.dtype == np.float32 and result.ref.shape == (5, 71)
    np.testing.assert_array_equal(result.ref, expected_ref)
    assert np.float16(1 + 2**-11 + 2**-25) != np.float16(result.ref[0, 7])
    assert result.clips == (ClipInfo("first", 0, 2), ClipInfo("second", 2, 3))
    assert tuple(plan.path for plan in result.plans) == tuple(paths)
    if compact:
        assert result.bodies == {}
    else:
        for key, expected in expected_bodies.items():
            assert result.bodies[key].dtype == expected.dtype
            np.testing.assert_array_equal(result.bodies[key], expected)


def test_every_clip_is_inspected_before_destination_allocation(tmp_path, monkeypatch):
    first, broken = tmp_path / "first.npz", tmp_path / "broken.npz"
    _motion(first)
    broken.write_bytes(b"not an archive")
    real_empty, real_inspect = np.empty, loading.inspect_motion
    completed = []

    def inspect(path):
        result = real_inspect(path)
        completed.append(path)
        return result

    def empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and shape[-1] == 71:
            raise AssertionError("Destination allocated before all sources were admitted")
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(loading, "inspect_motion", inspect)
    monkeypatch.setattr(np, "empty", empty)
    with pytest.raises(zipfile.BadZipFile):
        load_npz_motions([first, broken], ("pelvis",))
    assert completed == [first]


def test_plan_preserves_parent_traversal_after_directory_symlink(tmp_path):
    logical, actual = tmp_path / "logical", tmp_path / "actual"
    logical.mkdir()
    (actual / "nested").mkdir(parents=True)
    (logical / "link").symlink_to(actual / "nested", target_is_directory=True)
    _motion(logical / "clip.npz", frames=7)
    _motion(actual / "clip.npz", frames=3)
    path = logical / "link" / ".." / "clip.npz"

    plan = inspect_motion(path)

    assert plan.path == path.absolute() and plan.info["frames"] == 3
    with open_motion(path, plan=plan) as (data, info):
        assert info["frames"] == 3 and data["joint_pos"].shape == (3, 29)
    loaded = load_npz_motions([path], ("pelvis",))
    assert loaded.ref.shape == (3, 71) and loaded.clips == (ClipInfo("clip", 0, 3),)


def test_plan_has_only_immutable_metadata_and_reuse_skips_value_scans(tmp_path, monkeypatch):
    path = tmp_path / "motion.npz"
    values = _motion(path)
    original_getitem = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def record(data, key):
        reads.append(key)
        return original_getitem(data, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", record)
    plan = inspect_motion(path)
    assert all(reads.count(key) == 1 for key in _FIELDS)
    assert plan.path == path and set(plan.array_dtypes) == set(_FIELDS)
    assert all(plan.array_dtypes[key] == values[key].dtype for key in _FIELDS)
    stat = path.stat()
    assert plan.identity == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with zipfile.ZipFile(path) as archive:
        assert plan.uncompressed_bytes == sum(entry.file_size for entry in archive.infolist())
    with pytest.raises(TypeError):
        plan.info["frames"] = 99
    with pytest.raises(TypeError):
        plan.array_dtypes["joint_pos"] = np.dtype(np.float16)
    reads.clear()
    with open_motion(path, plan=plan) as (data, info):
        assert reads == []
        assert info == validate_motion(path)
        info["body_names"].clear()
        assert len(plan.info["body_names"]) == 30
        np.testing.assert_array_equal(data["joint_pos"], values["joint_pos"])


def test_plan_rejects_new_publication_but_opened_old_inode_remains_readable(tmp_path, monkeypatch):
    path, replacement = tmp_path / "motion.npz", tmp_path / "replacement.npz"
    _motion(path, frames=2)
    _motion(replacement, frames=3)
    plan = inspect_motion(path)
    original_load = np.load

    def rotate(stream, **kwargs):
        replacement.replace(path)
        return original_load(stream, **kwargs)

    monkeypatch.setattr(schema.np, "load", rotate)
    with open_motion(path, plan=plan) as (data, info):
        assert info["frames"] == 2 and data["joint_pos"].shape == (2, 29)
    monkeypatch.setattr(schema.np, "load", original_load)
    with pytest.raises(RuntimeError, match="changed since inspection"):
        with open_motion(path, plan=plan):
            raise AssertionError("A new inode reused the old plan")


def test_loading_rejects_path_rotation_between_planning_and_fill(tmp_path, monkeypatch):
    path, replacement = tmp_path / "motion.npz", tmp_path / "replacement.npz"
    _motion(path)
    _motion(replacement, frames=3)
    original = loading.inspect_motion

    def rotate(path):
        plan = original(path)
        replacement.replace(path)
        return plan

    monkeypatch.setattr(loading, "inspect_motion", rotate)
    with pytest.raises(RuntimeError, match="changed since inspection"):
        load_npz_motions([path], ("pelvis",))


@pytest.mark.parametrize("consumer_error", [False, True])
def test_planned_context_closes_descriptors_and_archive_on_every_exit(
    tmp_path, monkeypatch, consumer_error
):
    path = tmp_path / "motion.npz"
    _motion(path)
    plan = inspect_motion(path)
    real_open, real_load = Path.open, np.load
    streams, archives = [], []

    def record_open(target, *args, **kwargs):
        stream = real_open(target, *args, **kwargs)
        if target == path:
            streams.append(stream)
        return stream

    def record_load(*args, **kwargs):
        data = real_load(*args, **kwargs)
        archives.append(data)
        return data

    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(schema.np, "load", record_load)

    def consume():
        with open_motion(path, plan=plan) as (data, _):
            assert data["joint_pos"].shape == (2, 29)
            if consumer_error:
                raise LookupError("consumer failed")

    if consumer_error:
        with pytest.raises(LookupError, match="consumer failed"):
            consume()
    else:
        consume()
    assert len(streams) == len(archives) == 1
    assert streams[0].closed and archives[0].zip is None


def test_planned_context_detects_in_place_change_on_exit(tmp_path):
    path = tmp_path / "motion.npz"
    _motion(path)
    plan = inspect_motion(path)
    with pytest.raises(RuntimeError, match="changed during reading"):
        with open_motion(path, plan=plan):
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def test_loading_releases_each_clips_arrays_before_opening_the_next(tmp_path, monkeypatch):
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for path in paths:
        _motion(path)
    original_open = loading.open_motion
    references = []

    class TrackArrays:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            value = self.data[key]
            references.append(weakref.ref(value))
            return value

    @contextmanager
    def record(path, **kwargs):
        gc.collect()
        assert all(reference() is None for reference in references)
        with original_open(path, **kwargs) as (data, info):
            yield TrackArrays(data), info

    monkeypatch.setattr(loading, "open_motion", record)
    result = load_npz_motions(paths, ("pelvis",))
    assert result.ref.shape == (4, 71)
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    ("field", "shape", "descr"),
    [
        ("joint_pos", (10**12, 29), "<f8"),
        ("joint_names", (29,), "<U1000000000"),
        ("body_names", (10**12,), "<U0"),
        ("joint_pos", (2, 29), "|O"),
    ],
)
def test_malformed_npy_declarations_fail_before_any_array_load(
    tmp_path, monkeypatch, field, shape, descr
):
    path = tmp_path / "motion.npz"
    _motion(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"shape": shape, "fortran_order": False, "descr": descr}
    )
    entries[field + ".npy"] = header.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("Array allocation preceded header admission")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden)
    with pytest.raises(ValueError):
        inspect_motion(path)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_npy_versions_raw_alias_precedence_and_trailing_bytes_remain_supported(tmp_path, version):
    path = tmp_path / "motion.npz"
    values = _motion(path)
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in values.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, value, version=version, allow_pickle=False)
            # The raw member key must win over its otherwise valid .npy alias.
            archive.writestr(key, member.getvalue() + b"allowed trailing bytes")
            archive.writestr(key + ".npy", b"not the selected member")
    plan = inspect_motion(path)
    with open_motion(path, plan=plan) as (data, info):
        assert info["frames"] == 2
        np.testing.assert_array_equal(data["joint_pos"], values["joint_pos"])
