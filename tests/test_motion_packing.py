"""Packing preserves direct fp16 conversion without retaining every clip result."""

import gc
import importlib.util
import struct
import threading
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_motion_loading import _motion

from supermimic import resources
from supermimic.data import packing
from supermimic.data.packed import read_packed_motion
from supermimic.data.packing import pack_npz_motions
from supermimic.data.reference import ClipInfo


@pytest.fixture(autouse=True)
def _ample_ram(monkeypatch):
    monkeypatch.setattr(
        resources, "host_snapshot", lambda: {"ram_available_bytes": 100 * resources.GIB}
    )


def _direct_fp16_oracle(paths):
    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            root = data["body_names"].tolist().index("pelvis")
            chunks.append(np.concatenate([
                data["body_pos_w"][:, root], data["body_quat_w"][:, root],
                data["joint_pos"], data["body_lin_vel_w"][:, root],
                data["body_ang_vel_w"][:, root], data["joint_vel"],
            ], axis=1).astype(np.float16))
    return np.concatenate(chunks, axis=0)


def test_real_packing_preserves_clip_order_mixed_precision_and_direct_fp16(tmp_path):
    paths = [tmp_path / "z-last.npz", tmp_path / "a-first.npz"]
    _motion(paths[0], frames=2, dtypes={"body_pos_w": np.float32})
    _motion(paths[1], frames=3, order=range(29, -1, -1), offset=3)
    expected = _direct_fp16_oracle(paths)
    output = tmp_path / "packed.pt"
    increments = []

    packed = pack_npz_motions(paths, output, workers=2, progress=increments.append)
    restored = read_packed_motion(output)

    assert packed.clips == restored.clips == (
        ClipInfo("z-last", 0, 2), ClipInfo("a-first", 2, 3)
    )
    assert packed.ref.dtype == restored.ref.dtype == torch.float16
    assert packed.ref.numpy().tobytes() == restored.ref.numpy().tobytes() == expected.tobytes()
    assert packed.ref[0, 7].item() == np.float16(1 + 2**-11 + 2**-25)
    assert packed.ref[0, 7].item() != np.float16(np.float32(1 + 2**-11 + 2**-25))
    assert increments == [1, 1]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_pending_futures_and_result_arrays_never_exceed_worker_bound(tmp_path, monkeypatch, workers):
    paths = [tmp_path / f"clip-{index}.npz" for index in range(7)]
    for index, path in enumerate(paths):
        _motion(path, offset=index)
    future_refs, array_refs, submitted = [], [], []
    maximum = 0

    class CompletedFuture:
        def __init__(self, value):
            self.value = value
            array_refs.append(weakref.ref(value))

        def result(self):
            return self.value

        def cancel(self):
            return False

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, plan):
            nonlocal maximum
            gc.collect()
            live = sum(reference() is not None for reference in future_refs)
            assert live < workers
            assert sum(reference() is not None for reference in array_refs) <= live
            future = CompletedFuture(function(plan))
            future_refs.append(weakref.ref(future))
            submitted.append(plan.path)
            maximum = max(maximum, live + 1)
            return future

    monkeypatch.setattr(packing, "ThreadPoolExecutor", ImmediateExecutor)
    result = pack_npz_motions(paths, tmp_path / "packed.pt", workers=workers)

    assert submitted == paths and maximum == workers
    assert result.ref.shape == (14, 71)
    gc.collect()
    assert all(reference() is None for reference in future_refs + array_refs)
    np.testing.assert_array_equal(result.ref.numpy(), _direct_fp16_oracle(paths))


def test_worker_failure_waits_for_existing_worker_and_never_publishes(tmp_path, monkeypatch):
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for path in paths:
        _motion(path)
    original = packing._load_reference
    second_started, second_finished = threading.Event(), threading.Event()

    def fail_first(plan):
        if plan.path == paths[0]:
            assert second_started.wait(timeout=3)
            raise RuntimeError("clip decode failed")
        second_started.set()
        try:
            # Keep this worker alive while the first Future raises. The packer
            # must join it before propagating the failure to its caller.
            threading.Event().wait(timeout=0.02)
            return original(plan)
        finally:
            second_finished.set()

    monkeypatch.setattr(packing, "_load_reference", fail_first)
    output = tmp_path / "absent/packed.pt"
    with pytest.raises(RuntimeError, match="clip decode failed"):
        pack_npz_motions(paths, output, workers=2)
    assert second_finished.is_set()
    assert not output.exists() and not output.parent.exists()


def test_rotation_between_preflight_and_worker_load_is_not_published(tmp_path, monkeypatch):
    path, replacement = tmp_path / "clip.npz", tmp_path / "replacement.npz"
    _motion(path)
    _motion(replacement, frames=3)
    original = packing.inspect_motion

    def rotate(path):
        plan = original(path)
        replacement.replace(path)
        return plan

    monkeypatch.setattr(packing, "inspect_motion", rotate)
    output = tmp_path / "packed.pt"
    with pytest.raises(RuntimeError, match="changed since inspection"):
        pack_npz_motions([path], output, workers=1)
    assert not output.exists()


@pytest.mark.parametrize("reason", ["input_budget", "ram"])
def test_aggregate_budget_rejects_before_inspection_or_numpy_reads(tmp_path, monkeypatch, reason):
    path = tmp_path / "clip.npz"
    _motion(path)
    if reason == "input_budget":
        # Only the central directory is needed to reject this declared size;
        # neither the actual array payload nor its NPY header should be read.
        payload = bytearray(path.read_bytes())
        central = payload.index(b"PK\x01\x02")
        struct.pack_into("<I", payload, central + 24, 2 * resources.GIB + 1)
        path.write_bytes(payload)
    else:
        monkeypatch.setattr(resources, "host_snapshot", lambda: {"ram_available_bytes": 0})

    def forbidden(*args, **kwargs):
        raise AssertionError("Source values read before aggregate resource admission")

    monkeypatch.setattr(packing, "inspect_motion", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    output = tmp_path / "packed.pt"
    with pytest.raises(RuntimeError, match="2 GiB input or 8 GiB host RAM reserve"):
        pack_npz_motions([path], output)
    assert not output.exists()


@pytest.mark.parametrize("reason", ["input_budget", "ram"])
def test_original_resource_admission_still_precedes_workers(tmp_path, monkeypatch, reason):
    path = tmp_path / "clip.npz"
    _motion(path)
    if reason == "input_budget":
        plan = packing.inspect_motion(path)
        monkeypatch.setattr(packing, "inspect_motion", lambda path: replace(
            plan, uncompressed_bytes=2 * resources.GIB + 1
        ))
    else:
        monkeypatch.setattr(resources, "host_snapshot", lambda: {"ram_available_bytes": 0})

    def forbidden(plan):
        raise AssertionError("Worker started before resource admission")

    monkeypatch.setattr(packing, "_load_reference", forbidden)
    output = tmp_path / "packed.pt"
    with pytest.raises(RuntimeError, match="2 GiB input or 8 GiB host RAM reserve"):
        pack_npz_motions([path], output)
    assert not output.exists()


def test_existing_output_and_invalid_workers_are_not_changed(tmp_path):
    output = tmp_path / "packed.pt"
    output.write_bytes(b"existing publication")
    with pytest.raises(FileExistsError):
        pack_npz_motions([], output)
    for workers in (0, 5):
        with pytest.raises(ValueError, match="one and four"):
            pack_npz_motions([], output, workers=workers)
    assert output.read_bytes() == b"existing publication"


def test_script_retains_manifest_selection_and_sorted_clip_names(tmp_path):
    motions = tmp_path / "motions"
    motions.mkdir()
    for name in ("z", "a", "ignored"):
        _motion(motions / f"{name}.npz")
    manifest = tmp_path / "selected.csv"
    manifest.write_text("name,keep\nz,1\nignored,0\na,1\n")
    script = Path(__file__).resolve().parents[1] / "scripts/data/build_library.py"
    spec = importlib.util.spec_from_file_location("bounded_motion_packer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "packed.pt"

    module._pack(motions, output, manifest=manifest, workers=1)

    packed = read_packed_motion(output)
    assert packed.clips == (ClipInfo("a", 0, 2), ClipInfo("z", 2, 2))


def test_script_closes_progress_when_packing_raises(tmp_path, monkeypatch):
    import tqdm

    path = tmp_path / "clip.npz"
    _motion(path)
    script = Path(__file__).resolve().parents[1] / "scripts/data/build_library.py"
    spec = importlib.util.spec_from_file_location("failing_motion_packer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = {"increments": []}

    class Progress:
        def __init__(self, *, total):
            state["total"] = total

        def __enter__(self):
            return self

        def update(self, increment):
            state["increments"].append(increment)

        def __exit__(self, kind, value, traceback):
            state["closed_with"] = kind
            return False

    def fail(files, out, *, workers, progress):
        progress(1)
        raise RuntimeError("packing failed")

    monkeypatch.setattr(tqdm, "tqdm", Progress)
    monkeypatch.setattr(packing, "pack_npz_motions", fail)
    output = tmp_path / "packed.pt"
    with pytest.raises(RuntimeError, match="packing failed"):
        module._pack(tmp_path, output)
    assert state == {"total": 1, "increments": [1], "closed_with": RuntimeError}
    assert not output.exists()
