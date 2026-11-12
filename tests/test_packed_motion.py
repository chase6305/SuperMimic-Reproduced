"""Compact archive admission precedes device allocation and preserves legacy storage."""

import numpy as np
import pytest
import torch

from supermimic.data import packed
from supermimic.data.packed import read_packed_motion, write_packed_motion
from supermimic.data.reference import REF_DIM, REF_SLICES, REF_TERMS, ClipInfo, reference_array
from supermimic.data.schema import open_motion, validate_motion
from supermimic.interfaces import G1_BODY_NAMES, G1_JOINT_NAMES, QUATERNION_ORDER, ROBOT_ID
from supermimic.torch_io import read_torch_archive


def _reference(rows=3, dtype=torch.float32):
    ref = torch.zeros(rows, REF_DIM, dtype=dtype)
    ref[:, REF_SLICES["root_quat_w"]] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return ref


def _legacy(rows=3):
    return {"ref": _reference(rows), "names": ["clip"], "starts": [0], "lengths": [rows]}


def _read(tmp_path, payload, *, modern=True):
    path = tmp_path / "motion.pt"
    torch.save(payload, path, _use_new_zipfile_serialization=modern)
    return read_packed_motion(path)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_roundtrip_preserves_cpu_reference_dtype_and_clip_layout(tmp_path, dtype):
    ref = _reference(dtype=dtype)
    ref[:, REF_SLICES["root_quat_w"]] = torch.tensor([2**-0.5, 0, 0, 2**-0.5], dtype=dtype)
    clips = (ClipInfo("same-name", 0, 1), ClipInfo("same-name", 1, 2))
    path = tmp_path / "motion.pt"

    write_packed_motion(path, ref, clips)
    restored = read_packed_motion(path)

    assert restored.ref.device.type == "cpu" and restored.ref.dtype == dtype
    assert restored.clips == clips  # Duplicate names and a single-frame clip remain valid.
    torch.testing.assert_close(restored.ref, ref, rtol=0, atol=0)
    metadata = read_torch_archive(path).data["motion_format"]
    assert metadata == {
        "version": 1,
        "robot": ROBOT_ID,
        "quaternion_order": QUATERNION_ORDER,
        "fps": 50.0,
        "root_linear_velocity_frame": "world",
        "root_angular_velocity_frame": "world",
        "reference_terms": [list(term) for term in REF_TERMS],
    }


@pytest.mark.parametrize("modern", [False, True])
def test_legacy_four_key_archives_support_both_torch_serializations(tmp_path, modern):
    payload = _legacy()
    restored = _read(tmp_path, payload, modern=modern)
    assert restored.clips == (ClipInfo("clip", 0, 3),)
    torch.testing.assert_close(restored.ref, payload["ref"])


def test_npz_quaternion_boundary_survives_fp16_packing_without_changing_values(tmp_path):
    frames, bodies = 2, len(G1_BODY_NAMES)
    quaternions = np.zeros((frames, bodies, 4), dtype=np.float64)
    quaternions[..., 0] = 1
    quaternions[:, G1_BODY_NAMES.index("pelvis")] = [0.57849, 0.57849, 0.57849, 0]
    source = tmp_path / "boundary.npz"
    np.savez(
        source,
        fps=np.array([50.0]),
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(G1_BODY_NAMES),
        joint_pos=np.arange(frames * 29, dtype=np.float64).reshape(frames, 29) / 100,
        joint_vel=np.zeros((frames, 29), dtype=np.float64),
        body_pos_w=np.zeros((frames, bodies, 3), dtype=np.float64),
        body_quat_w=quaternions,
        body_lin_vel_w=np.zeros((frames, bodies, 3), dtype=np.float64),
        body_ang_vel_w=np.zeros((frames, bodies, 3), dtype=np.float64),
    )
    assert validate_motion(source)["frames"] == frames
    with open_motion(source) as (data, info):
        reference = reference_array(data, info["body_names"]).astype(np.float16)
    # The unchanged packer conversion pushes this admitted NPZ beyond the
    # original norm tolerance. Validation must allow its storage rounding.
    norm = np.linalg.norm(reference[:, REF_SLICES["root_quat_w"]].astype(np.float64), axis=-1)
    assert np.all(norm > 1 + 2e-3 + 1e-5)
    ref = torch.from_numpy(reference)
    clips = (ClipInfo("boundary", 0, frames),)
    new_path = tmp_path / "new.pt"
    write_packed_motion(new_path, ref, clips)
    paths = [new_path]
    for modern in (False, True):
        path = tmp_path / f"legacy-{modern}.pt"
        torch.save(
            {"ref": ref, "names": ["boundary"], "starts": [0], "lengths": [frames]},
            path,
            _use_new_zipfile_serialization=modern,
        )
        paths.append(path)
    for path in paths:
        restored = read_packed_motion(path)
        assert restored.clips == clips
        assert restored.ref.dtype == torch.float16
        assert restored.ref.numpy().tobytes() == reference.tobytes()
    assert ref.numpy().tobytes() == reference.tobytes()


@pytest.mark.parametrize("invalid", ["tables", "width"])
def test_library_rejects_invalid_cache_before_any_device_transfer(tmp_path, monkeypatch, invalid):
    from supermimic.data.motion_library import MotionLibrary

    payload = _legacy()
    if invalid == "tables":
        payload["lengths"] = [1, 2]
    else:
        payload["ref"] = payload["ref"][:, :-1]
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    q0 = torch.zeros(29)
    transfers = []

    def unexpected_transfer(self, *args, **kwargs):
        transfers.append((args, kwargs))
        raise AssertionError("Invalid compact data reached a device transfer")

    monkeypatch.setattr(torch.Tensor, "to", unexpected_transfer)
    with pytest.raises(ValueError, match="Packed motion"):
        MotionLibrary([path], ("pelvis",), q0, device="cuda", compact=True)
    assert transfers == []


def test_noncontiguous_dense_reference_is_preserved(tmp_path):
    payload = _legacy()
    storage = torch.zeros(3, REF_DIM * 2)
    payload["ref"] = storage[:, ::2]
    payload["ref"][:, REF_SLICES["root_quat_w"]] = torch.tensor([1.0, 0, 0, 0])
    restored = _read(tmp_path, payload)
    torch.testing.assert_close(restored.ref, payload["ref"])


@pytest.mark.parametrize(
    "update",
    [
        {"names": []},
        {"starts": []},
        {"lengths": [1, 2]},
        {"names": ["a", "b"], "starts": [0], "lengths": [1, 2]},
        {"starts": [1]},
        {"lengths": [0]},
        {"lengths": [-3]},
        {"lengths": [2]},
        {"lengths": [4]},
        {"names": ["a", "b"], "starts": [0, 0], "lengths": [1, 2]},
        {"names": ["a", "b"], "starts": [0, 2], "lengths": [1, 1]},
        {"names": [11]},
        {"names": "clip"},
    ],
)
def test_inconsistent_clip_tables_fail_without_silent_truncation(tmp_path, update):
    with pytest.raises(ValueError, match="Packed motion"):
        _read(tmp_path, {**_legacy(), **update})


@pytest.mark.parametrize("bad", [True, False, 0.0, 0.5, np.bool_(True), torch.tensor(True)])
def test_indices_reject_booleans_and_noninteger_values(tmp_path, bad):
    with pytest.raises(ValueError, match="must be an integer"):
        _read(tmp_path, {**_legacy(), "starts": [bad]})


def test_numpy_and_torch_integer_scalars_are_valid_table_indices(tmp_path):
    payload = _legacy()
    payload.update(
        names=["a", "b"],
        starts=[np.int64(0), torch.tensor(1, dtype=torch.int64)],
        lengths=[torch.tensor(1, dtype=torch.int32), np.int32(2)],
    )
    restored = _read(tmp_path, payload)
    assert restored.clips == (ClipInfo("a", 0, 1), ClipInfo("b", 1, 2))
    assert all(type(clip.start) is type(clip.length) is int for clip in restored.clips)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_references_are_rejected_in_every_chunk(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(packed, "_VALIDATION_ROWS", 2)
    payload = _legacy(rows=5)
    payload["ref"][-1, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        _read(tmp_path, payload)


def test_finite_validation_limits_the_size_of_each_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(packed, "_VALIDATION_ROWS", 2)
    original = torch.isfinite
    rows = []

    def record_finite(value):
        rows.append(value.shape[0])
        return original(value)

    monkeypatch.setattr(torch, "isfinite", record_finite)
    restored = _read(tmp_path, _legacy(rows=5))
    assert restored.ref.shape == (5, REF_DIM)
    assert rows and max(rows) <= 2


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(0, REF_DIM),
        torch.zeros(3, REF_DIM - 1),
        torch.zeros(3, REF_DIM, 1),
        torch.zeros(3, REF_DIM, dtype=torch.int64),
        torch.zeros(3, REF_DIM, dtype=torch.complex64),
        [[0.0] * REF_DIM] * 3,
        torch.zeros(3, REF_DIM).to_sparse(),
    ],
)
def test_invalid_reference_storage_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="dense real floating CPU tensor"):
        _read(tmp_path, {**_legacy(), "ref": bad})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1.02, 2])
def test_nonunit_root_quaternions_are_rejected(tmp_path, dtype, scale):
    payload = _legacy()
    payload["ref"] = _reference(dtype=dtype)
    payload["ref"][1, REF_SLICES["root_quat_w"]] *= scale
    with pytest.raises(ValueError, match="unit wxyz"):
        _read(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("version", 2),
        ("version", True),
        ("robot", "other-robot"),
        ("quaternion_order", "xyzw"),
        ("fps", 30),
        ("root_linear_velocity_frame", "body"),
        ("root_angular_velocity_frame", "body"),
        ("reference_terms", [["wrong", REF_DIM]]),
    ],
)
def test_explicit_incompatible_metadata_is_rejected(tmp_path, field, bad):
    path = tmp_path / "valid.pt"
    write_packed_motion(path, _reference(), [ClipInfo("clip", 0, 3)])
    payload = read_torch_archive(path).data
    payload["motion_format"][field] = bad
    with pytest.raises(ValueError, match="[Pp]acked motion"):
        _read(tmp_path, payload)


@pytest.mark.parametrize("metadata", [None, {}, "v1"])
def test_present_but_incomplete_metadata_is_not_treated_as_legacy(tmp_path, metadata):
    with pytest.raises(ValueError, match="Packed motion"):
        _read(tmp_path, {**_legacy(), "motion_format": metadata})


def test_unknown_extra_fields_do_not_change_reference_interpretation(tmp_path):
    payload = _legacy()
    payload["extra"] = {"origin": "local experiment"}
    restored = _read(tmp_path, payload)
    torch.testing.assert_close(restored.ref, payload["ref"])


def test_writer_validates_before_exclusive_publication(tmp_path):
    path = tmp_path / "motion.pt"
    ref, clips = _reference(), (ClipInfo("clip", 0, 3),)
    write_packed_motion(path, ref, clips)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_packed_motion(path, ref, clips)
    assert path.read_bytes() == original
    invalid = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match="contiguous"):
        write_packed_motion(invalid, ref, [ClipInfo("clip", 1, 3)])
    assert not invalid.exists()
    assert sorted(tmp_path.iterdir()) == [path]
