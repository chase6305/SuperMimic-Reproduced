"""Finite source arrays must remain finite along their actual storage casts."""

import sys
import warnings
import weakref
from types import ModuleType

import numpy as np
import pytest
import torch
from test_motion_loading import _motion, _old_concatenate_oracle

from supermimic.data import loading, motion_library
from supermimic.data.loading import load_npz_motions
from supermimic.data.motion_library import MotionLibrary
from supermimic.data.packed import write_packed_motion
from supermimic.data.reference import REF_DIM, ClipInfo
from supermimic.data.schema import validate_motion
from supermimic.interfaces import G1_BODY_NAMES


@pytest.fixture
def _query_only_fk(monkeypatch):
    class QueryOnlyFK:
        def __init__(self, model, *, device, dtype):
            assert device == torch.device("cpu")

        def body_indices(self, names):
            return torch.arange(len(names))

        def __call__(self, *args):
            raise AssertionError("Storage tests must not execute FK")

    converter = ModuleType("supermimic.data.motion_convert")
    converter.load_g1_model = object
    fk = ModuleType("supermimic.kinematics.g1_fk")
    fk.G1Kinematics = QueryOnlyFK
    monkeypatch.setitem(sys.modules, converter.__name__, converter)
    monkeypatch.setitem(sys.modules, fk.__name__, fk)


def _forbid_gpu(monkeypatch):
    real_to, real_as_tensor = torch.Tensor.to, torch.as_tensor
    gpu_calls = []

    def check(device):
        if device is not None and torch.device(device).type == "cuda":
            gpu_calls.append(device)
            raise AssertionError("GPU transfer preceded target storage validation")

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        check(device)
        return real_to(self, *args, **kwargs)

    def as_tensor(*args, **kwargs):
        check(kwargs.get("device"))
        return real_as_tensor(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    return gpu_calls


def _source(tmp_path, kind, value):
    path = tmp_path / ("motion.npz" if kind == "npz" else "motion.pt")
    if kind == "npz":
        arrays = _motion(path)
        arrays["body_pos_w"][:, 0, 0] = value
        np.savez(path, **arrays)
        expected = torch.from_numpy(_old_concatenate_oracle([path], ("pelvis",))[0])
    else:
        expected = torch.zeros(2, REF_DIM)
        expected[:, 3] = 1
        expected[:, 0] = value
        write_packed_motion(path, expected, (ClipInfo("motion", 0, 2),))
    return path, expected


def test_float64_source_overflow_at_float32_intermediate_is_rejected_in_blocks(
    tmp_path, monkeypatch
):
    path = tmp_path / "motion.npz"
    arrays = _motion(path, frames=5)
    arrays["joint_pos"][-1, -1] = 1e300
    np.savez(path, **arrays)
    assert validate_motion(path)["frames"] == 5
    monkeypatch.setattr(loading, "_VALIDATION_ROWS", 2)
    real_finite = np.isfinite
    rows = []

    def finite(value, *args, **kwargs):
        if getattr(value, "ndim", None) == 2 and value.shape[1] == REF_DIM:
            rows.append(len(value))
        return real_finite(value, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", finite)
    # Only the known overflow/invalid warnings from this conversion are muted;
    # rejection is explicit and unrelated runtime warnings remain errors.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="after float32 conversion"):
            load_npz_motions([path], ("pelvis",))
    assert rows == [2, 2, 1]


def test_float32_intermediate_overflow_fails_before_any_gpu_call(tmp_path, monkeypatch):
    path = tmp_path / "motion.npz"
    arrays = _motion(path)
    arrays["body_pos_w"][-1, 0, 0] = 1e300
    np.savez(path, **arrays)
    q0 = torch.zeros(29)
    calls = _forbid_gpu(monkeypatch)
    # A later float64 target cannot recover an overflowed float32 intermediate.
    with pytest.raises(ValueError, match="after float32 conversion"):
        MotionLibrary([path], ("pelvis",), q0, device="cuda", dtype=torch.float64)
    assert calls == []


@pytest.mark.parametrize("kind", ["npz", "packed"])
@pytest.mark.parametrize("value", [70000.0, -70000.0])
def test_compact_target_overflow_is_rejected_before_any_gpu_call(
    tmp_path, monkeypatch, kind, value
):
    path, _ = _source(tmp_path, kind, value)
    q0 = torch.zeros(29)
    calls = _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError, match="after conversion to torch.float16"):
        MotionLibrary(
            [path], ("pelvis",), q0, device="cuda", compact=True, store_dtype=torch.float16
        )
    assert calls == []


@pytest.mark.parametrize("kind", ["npz", "packed"])
def test_compact_query_overflow_is_rejected_before_any_gpu_call(tmp_path, monkeypatch, kind):
    path, _ = _source(tmp_path, kind, 70000.0)
    q0 = torch.zeros(29)
    calls = _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError, match="after query conversion to torch.float16"):
        MotionLibrary(
            [path], ("pelvis",), q0, device="cuda", compact=True,
            store_dtype=torch.float32, dtype=torch.float16,
        )
    assert calls == []


def test_packed_float64_storage_cannot_hide_float32_query_overflow(tmp_path, monkeypatch):
    path = tmp_path / "motion.pt"
    reference = torch.zeros(2, REF_DIM, dtype=torch.float64)
    reference[:, 3] = 1
    reference[:, 0] = 1e300
    write_packed_motion(path, reference, (ClipInfo("motion", 0, 2),))
    q0 = torch.zeros(29)
    calls = _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError, match="after query conversion to torch.float32"):
        MotionLibrary(
            [path], ("pelvis",), q0, device="cuda", compact=True,
            store_dtype=torch.float64, dtype=torch.float32,
        )
    assert calls == []


@pytest.mark.usefixtures("_query_only_fk")
@pytest.mark.parametrize("kind", ["npz", "packed"])
def test_larger_finite_values_remain_valid_for_float32_storage(tmp_path, kind):
    path, expected = _source(tmp_path, kind, 70000.0)
    library = MotionLibrary(
        [path], ("pelvis",), torch.zeros(29), device="cpu", compact=True, store_dtype=torch.float32
    )
    assert library._ref.dtype == torch.float32
    torch.testing.assert_close(library._ref, expected, rtol=0, atol=0)


@pytest.mark.usefixtures("_query_only_fk")
@pytest.mark.parametrize("kind", ["npz", "packed"])
@pytest.mark.parametrize("value", [65519.0, -65519.0])
def test_values_above_half_max_that_round_finite_preserve_the_original_cast(tmp_path, kind, value):
    path, expected = _source(tmp_path, kind, value)
    original = expected.clone()
    library = MotionLibrary(
        [path], ("pelvis",), torch.zeros(29), device="cpu", compact=True, store_dtype=torch.float16
    )
    assert library._ref[0, 0].item() == (65504.0 if value > 0 else -65504.0)
    torch.testing.assert_close(library._ref, expected.to(torch.float16), rtol=0, atol=0)
    torch.testing.assert_close(expected, original, rtol=0, atol=0)


@pytest.mark.usefixtures("_query_only_fk")
@pytest.mark.parametrize("kind", ["npz", "packed"])
def test_finite_rounding_is_preserved_when_query_precision_is_narrower_than_storage(tmp_path, kind):
    path, expected = _source(tmp_path, kind, 65519.0)
    library = MotionLibrary(
        [path], ("pelvis",), torch.zeros(29), device="cpu", compact=True,
        store_dtype=torch.float32, dtype=torch.float16,
    )
    torch.testing.assert_close(library._ref, expected, rtol=0, atol=0)
    queried = library.ref71(torch.tensor([0, 1]))
    assert queried.dtype == torch.float16 and queried[0, 0].item() == 65504.0
    torch.testing.assert_close(queried, expected.to(torch.float16), rtol=0, atol=0)


def test_query_preview_uses_actual_storage_intermediate_at_double_rounding_boundary(monkeypatch):
    source = torch.tensor([[65519.999]], dtype=torch.float64)
    # Direct source->half remains finite, whereas the float32 storage rounding
    # raises it to 65520, which then overflows the half query dtype.
    assert source.numpy().astype(np.float16).item() == 65504.0
    real_to = torch.Tensor.to
    casts = []

    def record(self, *args, **kwargs):
        assert kwargs.get("device") == "cpu"
        casts.append((self.dtype, kwargs["dtype"], self.item()))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record)
    with pytest.raises(ValueError, match="after query conversion to torch.float16"):
        motion_library._validate_storage_dtype(
            source, torch.float32, "reference", query_dtype=torch.float16
        )
    assert casts == [
        (torch.float64, torch.float32, 65519.999),
        (torch.float32, torch.float16, 65520.0),
    ]


@pytest.mark.parametrize("field", ["body_pos_w", "body_lin_vel_w", "body_ang_vel_w"])
def test_nonroot_body_target_overflow_is_checked_before_gpu_transfer(tmp_path, monkeypatch, field):
    path = tmp_path / "motion.npz"
    arrays = _motion(path)
    arrays[field][:, G1_BODY_NAMES.index("torso_link"), 0] = 1e300
    np.savez(path, **arrays)
    assert validate_motion(path)["frames"] == 2
    assert np.isfinite(load_npz_motions([path], ("pelvis", "torso_link")).ref).all()
    q0 = torch.zeros(29)
    with monkeypatch.context() as gpu_guard:
        calls = _forbid_gpu(gpu_guard)
        with pytest.raises(ValueError, match="Motion body .*after conversion to torch.float32"):
            MotionLibrary([path], ("pelvis", "torso_link"), q0, device="cuda")
        assert calls == []
    # Source values stay valid when the requested body storage can represent them.
    library = MotionLibrary(
        [path], ("pelvis", "torso_link"), q0, device="cpu", dtype=torch.float64
    )
    assert getattr(library, field)[0, 1, 0].item() == 1e300


def test_target_cast_preview_is_bounded_and_catches_overflow_in_the_last_block(monkeypatch):
    source = torch.zeros(5, REF_DIM)
    source[:, 3] = 1
    source[:, 0] = 65519
    source[-1, 0] = 70000
    original = source.clone()
    monkeypatch.setattr(motion_library, "_VALIDATION_ROWS", 2)
    real_to, real_finite = torch.Tensor.to, torch.isfinite
    rows, previews, scans = [], [], []

    def record(self, *args, **kwargs):
        assert kwargs.get("device") == "cpu" and kwargs.get("dtype") == torch.float16
        assert all(reference() is None for reference in previews + scans)
        rows.append(len(self))
        result = real_to(self, *args, **kwargs)
        previews.append(weakref.ref(result))
        return result

    def finite(value):
        assert value.dtype == torch.float32 and len(value) <= 2
        scans.append(weakref.ref(value))
        return real_finite(value)

    monkeypatch.setattr(torch.Tensor, "to", record)
    monkeypatch.setattr(torch, "isfinite", finite)
    with pytest.raises(ValueError, match="after conversion to torch.float16"):
        motion_library._validate_storage_dtype(source, torch.float16, "reference")
    assert rows == [2, 2, 1]
    assert torch.equal(source, original)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="needs float8_e4m3fn casts")
@pytest.mark.parametrize("overflow", [False, True])
def test_float8_target_preview_uses_supported_float32_finite_scan(monkeypatch, overflow):
    source = torch.tensor([[0.0, 1.0, -2.0], [448.0, 10.0, -10.0]])
    if overflow:
        source[-1, 0] = 70000.0
    original = source.clone()
    real_finite = torch.isfinite
    scans = []

    def finite(value):
        # The actual CPU conversion is float8, while only this finite-value
        # check is widened. Calling isfinite directly on the shell dtype would
        # require a kernel that the original storage path never needed.
        assert value.dtype == torch.float32
        scans.append(value.shape)
        return real_finite(value)

    monkeypatch.setattr(torch, "isfinite", finite)
    if overflow:
        with pytest.raises(ValueError, match="after conversion to torch.float8_e4m3fn"):
            motion_library._validate_storage_dtype(source, torch.float8_e4m3fn, "reference")
    else:
        motion_library._validate_storage_dtype(source, torch.float8_e4m3fn, "reference")
    assert scans == [source.shape]
    assert torch.equal(source, original)
