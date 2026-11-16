"""Trajectory admission and policy checks use the same real NPZ publication."""

import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _write_trajectory(path, *, steps=2, labels=True, contacts_extra=0, **changes):
    import numpy as np

    from supermimic.deployment.trajectory import PHYSICAL_CONTACT_COLUMNS

    contacts = np.zeros((steps * 4 + contacts_extra, 13))
    contacts[:, 0] = np.arange(1, len(contacts) + 1) * 0.005
    arrays = {
        "states": np.full((steps + 1, 71), float(steps)),
        "control_dt": np.array(0.02),
        "actions": np.zeros((steps, 29), dtype=np.float32),
        "reference_frame": np.minimum(np.arange(steps) + 1, 2),
        "physical_contacts": contacts,
    }
    if labels:
        arrays["physical_contact_columns"] = np.array(PHYSICAL_CONTACT_COLUMNS)
    arrays.update(changes)
    np.savez(path, **arrays)
    return arrays


def test_complete_trajectory_preserves_arrays_and_source_fingerprint(tmp_path):
    import numpy as np

    from supermimic.deployment.trajectory import (
        PHYSICAL_CONTACT_COLUMNS,
        open_trajectory,
        validate_reference_clock,
        validate_transitions,
    )

    path = tmp_path / "trajectory.npz"
    expected = _write_trajectory(path)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with open_trajectory(
        path, fields=("actions", "reference_frame", "physical_contacts")
    ) as record:
        assert record.sha256 == expected_sha
        assert record.control_dt == 0.02
        assert record.states.shape == (3, 71)
        assert record.arrays["actions"].dtype == np.float32
        assert record.contact_layout_verified is True
        assert tuple(record.arrays["physical_contact_columns"]) == PHYSICAL_CONTACT_COLUMNS
        for name, value in record.arrays.items():
            np.testing.assert_array_equal(value, expected[name])
        validate_transitions(record)
        validate_reference_clock(record, motion_frames=3, offset=1)
        with pytest.raises(TypeError):
            record.arrays["new"] = np.array(1)
    # Loaded arrays have no deferred archive reads after the context closes.
    np.testing.assert_array_equal(record.states, expected["states"])


def test_base_reader_allows_states_only_and_does_not_load_unrequested_fields(tmp_path):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory, validate_transitions

    path = tmp_path / "states.npz"
    np.savez(
        path,
        states=np.zeros((1, 3)),
        control_dt=np.array(0.02),
        # Loading these with allow_pickle=False or validating them would fail.
        unused=np.array([{"not": "trajectory evidence"}], dtype=object),
        physical_contacts=np.array(["not requested"]),
    )
    with open_trajectory(path) as record:
        assert set(record.arrays) == {"states", "control_dt"}
        assert record.states.shape == (1, 3)
        assert record.contact_layout_verified is None
        with pytest.raises(ValueError, match="Missing trajectory field: actions"):
            validate_transitions(record)
    with pytest.raises(ValueError, match="Missing trajectory fields: actions"):
        with open_trajectory(path, fields=("actions",)):
            pytest.fail("Missing requested evidence must not yield a record")


def test_single_initial_state_is_a_valid_zero_transition_record(tmp_path):
    from supermimic.deployment.trajectory import (
        open_trajectory,
        validate_reference_clock,
        validate_transitions,
    )

    path = tmp_path / "initial.npz"
    _write_trajectory(path, steps=0)
    with open_trajectory(path, fields=("actions", "reference_frame")) as record:
        validate_transitions(record)
        validate_reference_clock(record, motion_frames=1, offset=1)
        assert record.states.shape == (1, 71)


@pytest.mark.parametrize("missing", ["states", "control_dt"])
def test_base_fields_are_required_even_with_an_empty_selection(tmp_path, missing):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory

    path = tmp_path / "missing-base.npz"
    arrays = {"states": np.zeros((1, 3)), "control_dt": np.array(0.02)}
    del arrays[missing]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Missing trajectory fields: " + missing):
        with open_trajectory(path, fields=()):
            pytest.fail("States and control_dt must always be present")


def test_explicit_legacy_clock_fallback_preserves_archive_identity_and_budget(tmp_path):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory, validate_transitions

    path = tmp_path / "legacy.npz"
    arrays = _write_trajectory(path)
    del arrays["control_dt"]
    np.savez(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        budget = sum(entry.file_size for entry in archive.infolist())

    with open_trajectory(
        path, fields=("actions", "reference_frame", "physical_contacts"),
        fallback_control_dt=0.02, byte_budget=budget,
    ) as record:
        assert record.sha256 == digest
        assert record.control_dt == 0.02 and record.control_dt_source == "fallback"
        validate_transitions(record)


def test_explicit_fallback_never_overwrites_a_recorded_clock(tmp_path):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory

    path = tmp_path / "recorded.npz"
    _write_trajectory(path, control_dt=np.array(0.04))
    with open_trajectory(path, fallback_control_dt=0.02) as record:
        assert record.control_dt == 0.04 and record.control_dt_source == "trajectory"
    _write_trajectory(path, control_dt=np.array([0.02, 0.02]))
    with pytest.raises(ValueError, match="control_dt"):
        with open_trajectory(path, fallback_control_dt=0.02):
            pytest.fail("A fallback cannot repair an invalid recorded clock")


@pytest.mark.parametrize("fallback", [0, -1, float("nan"), float("inf"), True, "0.02"])
def test_invalid_fallback_clock_is_rejected_before_opening_a_file(tmp_path, fallback):
    from supermimic.deployment.trajectory import open_trajectory

    with pytest.raises(ValueError, match="Fallback control_dt"):
        with open_trajectory(tmp_path / "absent.npz", fallback_control_dt=fallback):
            pytest.fail("Invalid fallback must not admit a legacy trajectory")


@pytest.mark.parametrize(
    "case",
    [
        "states_1d",
        "states_empty",
        "states_nan",
        "states_text",
        "dt_vector",
        "dt_zero",
        "dt_nan",
        "dt_text",
    ],
)
def test_base_rejects_invalid_states_or_control_dt(tmp_path, case):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory

    bad = {
        "states_1d": {"states": np.zeros(71)},
        "states_empty": {"states": np.zeros((0, 71))},
        "states_nan": {"states": np.full((1, 71), np.nan)},
        "states_text": {"states": np.array([["0"]])},
        "dt_vector": {"control_dt": np.array([0.02, 0.02])},
        "dt_zero": {"control_dt": np.array(0.0)},
        "dt_nan": {"control_dt": np.array(np.nan)},
        "dt_text": {"control_dt": np.array("0.02")},
    }
    path = tmp_path / "bad.npz"
    _write_trajectory(path, **bad[case])
    with pytest.raises(ValueError, match="Trajectory (states|control_dt)"):
        with open_trajectory(path):
            pytest.fail("Invalid base arrays must not yield a record")


@pytest.mark.parametrize(
    "case",
    ["state_width", "action_width", "action_nan", "action_count", "frame_count"],
)
def test_transition_policy_rejects_invalid_alignment(tmp_path, case):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory, validate_transitions

    bad = {
        "state_width": {"states": np.zeros((3, 70))},
        "action_width": {"actions": np.zeros((2, 28))},
        "action_nan": {"actions": np.full((2, 29), np.nan)},
        "action_count": {"actions": np.zeros((3, 29))},
        "frame_count": {"reference_frame": np.array([1])},
    }
    path = tmp_path / "bad-transitions.npz"
    _write_trajectory(path, **bad[case])
    with open_trajectory(path, fields=("actions", "reference_frame")) as record:
        with pytest.raises(ValueError, match="Trajectory|Inconsistent"):
            validate_transitions(record)


def test_requested_frame_dtype_is_rejected_before_array_loading(tmp_path, monkeypatch):
    import numpy as np
    from numpy.lib.npyio import NpzFile

    from supermimic.deployment.trajectory import open_trajectory

    path = tmp_path / "float-frames.npz"
    _write_trajectory(path, reference_frame=np.array([1.0, 2.0]))

    def unexpected_array(*args, **kwargs):
        pytest.fail("All selected header checks must finish before any array is loaded")

    monkeypatch.setattr(NpzFile, "__getitem__", unexpected_array)
    with pytest.raises(ValueError, match="reference_frame.*integer"):
        with open_trajectory(path, fields=("actions", "reference_frame")):
            pytest.fail("Float frame indices must not yield a record")


def test_custom_state_and_action_widths_remain_explicit_policy(tmp_path):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory, validate_transitions

    path = tmp_path / "custom.npz"
    _write_trajectory(path, states=np.zeros((3, 3)), actions=np.zeros((2, 2)))
    with open_trajectory(path, fields=("actions", "reference_frame")) as record:
        validate_transitions(record, state_dim=3, action_dim=2)
        with pytest.raises(ValueError, match="71 columns"):
            validate_transitions(record)


@pytest.mark.parametrize(
    "field,shape,descr,message",
    [
        ("states", (10**9, 71), "<U0", "Trajectory states"),
        ("states", (1, 0), "<f8", "Trajectory states"),
        ("control_dt", (10**9,), "|S0", "Trajectory control_dt"),
        ("actions", (10**9, 29), "|S0", "Trajectory actions"),
        ("reference_frame", (10**9,), "<U0", "Trajectory reference_frame"),
        ("physical_contacts", (10**9, 13), "|S0", "Trajectory physical_contacts"),
        ("physical_contact_columns", (10**9,), "|S0", "Trajectory physical_contact_columns"),
        ("joint_names", (10**9,), "<U0", "Trajectory joint_names"),
        ("actions", (2**40, 29), "<f8", "payload exceeds member size"),
    ],
)
def test_selected_headers_reject_oversized_declarations_before_any_array_access(
    tmp_path, monkeypatch, field, shape, descr, message
):
    import numpy as np
    from numpy.lib.npyio import NpzFile

    from supermimic.deployment.trajectory import open_trajectory

    path = tmp_path / "declared.npz"
    _write_trajectory(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    # A genuine NPY header can declare huge shapes without allocating an array.
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"shape": shape, "descr": descr, "fortran_order": False}
    )
    members[field + ".npy"] = header.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, member in members.items():
            archive.writestr(name, member)

    def unexpected_array(*args, **kwargs):
        pytest.fail("A bad selected header must be rejected before even states are allocated")

    monkeypatch.setattr(NpzFile, "__getitem__", unexpected_array)
    with pytest.raises(ValueError, match=message):
        with open_trajectory(path, fields=(field,)):
            pytest.fail("Invalid headers must not yield a record")


def test_raw_member_priority_and_budget_match_the_array_numpy_will_load(tmp_path):
    import numpy as np

    from supermimic.deployment.trajectory import open_trajectory

    path = tmp_path / "aliases.npz"
    _write_trajectory(path)
    raw_states = np.array([[1.0, 2.0, 3.0]])
    raw_member = io.BytesIO()
    np.save(raw_member, raw_states, allow_pickle=False)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("states", raw_member.getvalue())
        budget = archive.getinfo("states").file_size + archive.getinfo("control_dt.npy").file_size
    with open_trajectory(path, byte_budget=budget) as record:
        np.testing.assert_array_equal(record.states, raw_states)
    # The lower-priority .npy member still exists and contains a different array.
    with np.load(path, allow_pickle=False) as data:
        assert data["states.npy"].shape == (3, 71)


@pytest.mark.parametrize("offset,frames", [(0, [0, 1, 2, 2]), (1, [1, 2, 2, 2]), (7, [2, 2, 2, 2])])
def test_reference_clock_clamps_at_the_last_motion_frame(tmp_path, offset, frames):
    import numpy as np

    from supermimic.deployment.trajectory import (
        open_trajectory,
        validate_reference_clock,
        validate_transitions,
    )

    path = tmp_path / "clamped.npz"
    _write_trajectory(path, steps=4, reference_frame=np.array(frames))
    with open_trajectory(path, fields=("actions", "reference_frame")) as record:
        validate_transitions(record)
        validate_reference_clock(record, motion_frames=3, offset=offset)


@pytest.mark.parametrize("motion_frames,offset", [(3, 0), (0, 1), (3, -1), (True, 1), (3, 1.0)])
def test_reference_clock_rejects_discontinuity_or_invalid_parameters(
    tmp_path, motion_frames, offset
):
    from supermimic.deployment.trajectory import open_trajectory, validate_reference_clock

    path = tmp_path / "clock.npz"
    _write_trajectory(path)
    with open_trajectory(path, fields=("reference_frame",)) as record:
        with pytest.raises(ValueError, match="Reference"):
            validate_reference_clock(record, motion_frames=motion_frames, offset=offset)


@pytest.mark.parametrize("labels", [False, True])
def test_partial_control_contacts_are_retained_without_claiming_missing_labels(tmp_path, labels):
    from supermimic.deployment.trajectory import open_trajectory, validate_transitions

    path = tmp_path / "partial.npz"
    _write_trajectory(path, contacts_extra=2, labels=labels)
    with open_trajectory(
        path, fields=("actions", "reference_frame", "physical_contacts")
    ) as record:
        validate_transitions(record)
        assert record.arrays["physical_contacts"].shape == (10, 13)
        assert record.contact_layout_verified is labels
        assert ("physical_contact_columns" in record.arrays) is labels


@pytest.mark.parametrize("case", ["width", "nan", "labels", "label_shape"])
def test_requested_contacts_reject_invalid_shape_or_explicit_layout(tmp_path, case):
    import numpy as np

    from supermimic.deployment.trajectory import PHYSICAL_CONTACT_COLUMNS, open_trajectory

    bad = {
        "width": {"physical_contacts": np.zeros((8, 12))},
        "nan": {"physical_contacts": np.full((8, 13), np.nan)},
        "labels": {"physical_contact_columns": np.array(PHYSICAL_CONTACT_COLUMNS[::-1])},
        "label_shape": {"physical_contact_columns": np.array([PHYSICAL_CONTACT_COLUMNS])},
    }
    path = tmp_path / "contacts.npz"
    _write_trajectory(path, **bad[case])
    with pytest.raises(ValueError, match="Trajectory physical_contact"):
        with open_trajectory(path, fields=("physical_contacts",)):
            pytest.fail("Invalid requested contact evidence must not yield a record")


def test_budget_counts_selected_members_and_rejects_before_array_loading(tmp_path, monkeypatch):
    import numpy as np

    from supermimic.deployment import trajectory

    path = tmp_path / "budget.npz"
    _write_trajectory(path, unused=np.zeros(10_000))
    with zipfile.ZipFile(path) as archive:
        budget = sum(archive.getinfo(name + ".npy").file_size for name in ("states", "control_dt"))
    with trajectory.open_trajectory(path, byte_budget=budget) as record:
        assert set(record.arrays) == {"states", "control_dt"}

    def unexpected_load(*args, **kwargs):
        pytest.fail("An over-budget trajectory must fail before np.load")

    monkeypatch.setattr(trajectory.np, "load", unexpected_load)
    with pytest.raises(ValueError, match="unpacked byte budget"):
        with trajectory.open_trajectory(path, byte_budget=budget - 1):
            pytest.fail("Over-budget arrays must not be yielded")


def test_atomic_replacement_keeps_loaded_arrays_and_hash_bound_to_open_inode(tmp_path, monkeypatch):
    import numpy as np

    from supermimic.deployment import trajectory

    path, replacement = tmp_path / "trajectory.npz", tmp_path / "replacement.npz"
    expected = _write_trajectory(path, steps=2)
    _write_trajectory(replacement, steps=4)
    old_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    real_load = np.load
    streams, archives = [], []

    def replace_before_loading(stream, **kwargs):
        # Real admission and the first source hash already used this descriptor.
        # A concurrent publisher now changes only the pathname, not its inode.
        replacement.replace(path)
        streams.append(stream)
        archives.append(real_load(stream, **kwargs))
        return archives[-1]

    monkeypatch.setattr(trajectory.np, "load", replace_before_loading)
    with trajectory.open_trajectory(path, fields=("actions", "reference_frame")) as record:
        assert record.sha256 == old_sha
        np.testing.assert_array_equal(record.states, expected["states"])
        assert not streams[0].closed and archives[0].zip is not None
    assert streams[0].closed and archives[0].zip is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() != old_sha
    with real_load(path, allow_pickle=False) as current:
        assert current["states"].shape == (5, 71)


def test_same_size_in_place_mutation_is_rejected_even_if_mtime_is_restored(tmp_path, monkeypatch):
    import numpy as np

    from supermimic.deployment import trajectory

    path = tmp_path / "trajectory.npz"
    _write_trajectory(path)
    before = path.stat()
    real_load = np.load
    streams, archives = [], []

    def mutate_after_archive_is_open(stream, **kwargs):
        data = real_load(stream, **kwargs)
        streams.append(stream)
        archives.append(data)
        # Alter the already-read ZIP end record, leaving array members intact.
        # Restoring mtime proves that the second content hash is also checked.
        with path.open("r+b", buffering=0) as writer:
            writer.seek(-1, os.SEEK_END)
            last = writer.read(1)
            writer.seek(-1, os.SEEK_END)
            writer.write(bytes([last[0] ^ 1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return data

    monkeypatch.setattr(trajectory.np, "load", mutate_after_archive_is_open)
    with pytest.raises(RuntimeError, match="changed in place"):
        with trajectory.open_trajectory(path):
            pytest.fail("Changed source bytes must not produce a fingerprinted record")
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert streams[0].closed and archives[0].zip is None


@pytest.mark.parametrize("failure", ["consumer", "validation", "missing", "budget", "archive"])
def test_all_exit_paths_close_the_owned_file_and_open_archive(tmp_path, monkeypatch, failure):
    import numpy as np

    from supermimic.deployment import trajectory

    path = tmp_path / "trajectory.npz"
    _write_trajectory(path, **({"control_dt": np.array(-1.0)} if failure == "validation" else {}))
    if failure == "archive":
        path.write_bytes(b"not a ZIP archive")
    real_open, real_load = Path.open, np.load
    streams, archives = [], []

    def capture_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        if self == path:
            streams.append(stream)
        return stream

    def capture_load(stream, **kwargs):
        data = real_load(stream, **kwargs)
        archives.append(data)
        return data

    monkeypatch.setattr(Path, "open", capture_open)
    monkeypatch.setattr(trajectory.np, "load", capture_load)
    options = {"fields": ("missing",)} if failure == "missing" else {}
    if failure == "budget":
        options["byte_budget"] = 0
    error = (
        RuntimeError
        if failure == "consumer"
        else zipfile.BadZipFile
        if failure == "archive"
        else ValueError
    )
    with pytest.raises(error):
        with trajectory.open_trajectory(path, **options):
            assert failure == "consumer"
            raise RuntimeError("consumer failed")
    assert len(streams) == 1 and streams[0].closed
    assert all(archive.zip is None for archive in archives)
    assert bool(archives) is (failure in ("consumer", "validation"))


def test_trajectory_import_does_not_load_policy_simulator_or_scipy():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "mujoco", "mujoco_warp", "mjlab", "warp", "scipy"}

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError(f"Trajectory reader imported runtime: {fullname}")

sys.meta_path.insert(0, NoRuntime())
from supermimic.deployment.trajectory import open_trajectory
assert callable(open_trajectory)
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
