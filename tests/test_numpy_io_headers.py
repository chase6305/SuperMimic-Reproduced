"""NPZ header admission checks the selected member before allocating arrays."""

import struct
import zipfile

import numpy as np
import pytest

from supermimic import numpy_io
from supermimic.numpy_io import inspect_npz_headers


def _header_text(shape, descr="<f4", *, fortran=False):
    return repr({"descr": descr, "fortran_order": fortran, "shape": shape}) + "\n"


def _member(shape=(2,), descr="<f4", *, version=(1, 0), payload=b"", fortran=False):
    encoding = "utf8" if version == (3, 0) else "latin1"
    header = _header_text(shape, descr, fortran=fortran).encode(encoding)
    size_format = "<H" if version == (1, 0) else "<I"
    return b"\x93NUMPY" + bytes(version) + struct.pack(size_format, len(header)) + header + payload


def _archive(tmp_path, members):
    path = tmp_path / "arrays.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _forbid_array_reads(monkeypatch, data):
    def unexpected_array(*args, **kwargs):
        raise AssertionError("Header inspection must not materialize NPZ arrays")

    monkeypatch.setattr(type(data), "__getitem__", unexpected_array)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_supported_versions_return_metadata_without_loading_arrays(tmp_path, monkeypatch, version):
    path = _archive(
        tmp_path,
        {"values.npy": _member((2, 3), version=version, payload=bytes(24), fortran=True)},
    )
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        header = inspect_npz_headers(data, ["values"])["values"]
        assert header.shape == (2, 3)
        assert header.dtype == np.dtype("<f4")
        assert header.fortran_order is True and header.nbytes == 24
        assert data.zip.fp is not None  # The caller still owns the archive.


@pytest.mark.parametrize(
    ("shape", "descr"),
    [((2**40,), "<f8"), ((1,), "<U100000000"), ((2**40, 2**40), "<f4")],
)
def test_large_declared_payloads_are_rejected_from_tiny_members(
    tmp_path, monkeypatch, shape, descr
):
    path = _archive(tmp_path, {"values.npy": _member(shape, descr)})
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        with pytest.raises(ValueError, match="payload exceeds member size"):
            inspect_npz_headers(data, ["values"])


@pytest.mark.parametrize(
    ("shape", "descr", "message"),
    [((1,), "|O", "Object dtype"), ((-2, 3), "<f4", "Negative shape")],
)
def test_object_and_negative_shape_headers_are_rejected(
    tmp_path, monkeypatch, shape, descr, message
):
    path = _archive(tmp_path, {"values.npy": _member(shape, descr)})
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        with pytest.raises(ValueError, match=message):
            inspect_npz_headers(data, ["values"])


@pytest.mark.parametrize("descr", ["<U0", "|S0"])
def test_zero_width_strings_keep_generic_header_semantics(tmp_path, monkeypatch, descr):
    path = _archive(tmp_path, {"values.npy": _member((10**9,), descr)})
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        header = inspect_npz_headers(data, ["values"])["values"]
        assert header.shape == (10**9,) and header.nbytes == 0


@pytest.mark.parametrize("invalid_bare", [False, True])
def test_exact_raw_member_precedes_npy_alias(tmp_path, monkeypatch, invalid_bare):
    valid = _member((2,), payload=bytes(8))
    invalid = _member((2**40,))
    path = _archive(
        tmp_path,
        {"values.npy": valid if invalid_bare else invalid, "values": invalid if invalid_bare else valid},
    )
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        if invalid_bare:
            with pytest.raises(ValueError, match="payload exceeds member size"):
                inspect_npz_headers(data, ["values"])
            assert inspect_npz_headers(data, ["values.npy"])["values.npy"].shape == (2,)
        else:
            assert inspect_npz_headers(data, ["values"])["values"].shape == (2,)


def test_trailing_member_bytes_are_allowed(tmp_path, monkeypatch):
    path = _archive(
        tmp_path, {"values.npy": _member((2,), "<f8", payload=bytes(16) + b"unused tail")}
    )
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        assert inspect_npz_headers(data, ["values"])["values"].nbytes == 16


def test_declared_header_larger_than_member_is_rejected_before_numpy_parser(tmp_path, monkeypatch):
    payload = b"\x93NUMPY\x02\x00" + struct.pack("<I", 2**31) + b"{}"
    path = _archive(tmp_path, {"values.npy": payload})

    def unexpected_parser(*args, **kwargs):
        raise AssertionError("An unbounded header must not reach NumPy's reader")

    monkeypatch.setattr(numpy_io, "_read_array_header", unexpected_parser)
    with np.load(path, allow_pickle=False) as data:
        with pytest.raises(ValueError, match="header exceeds member size"):
            inspect_npz_headers(data, ["values"])


@pytest.mark.parametrize(("version", "multiplier"), [((1, 0), 1), ((2, 0), 1), ((3, 0), 4)])
def test_header_encoding_bound_is_checked_before_numpy_parser(
    tmp_path, monkeypatch, version, multiplier
):
    limit = 20
    header = b"x" * (limit * multiplier + 1)
    size_format = "<H" if version == (1, 0) else "<I"
    payload = b"\x93NUMPY" + bytes(version) + struct.pack(size_format, len(header)) + header
    path = _archive(tmp_path, {"values.npy": payload})

    def unexpected_parser(*args, **kwargs):
        raise AssertionError("An oversized header must not reach NumPy's reader")

    monkeypatch.setattr(numpy_io, "_read_array_header", unexpected_parser)
    with np.load(path, allow_pickle=False, max_header_size=limit) as data:
        with pytest.raises(ValueError, match="header exceeds header budget"):
            inspect_npz_headers(data, ["values"])


def test_utf8_headers_use_decoded_character_limit_not_raw_byte_limit(tmp_path, monkeypatch):
    field_name = "机器人" * 20
    descr = [(field_name, "<i4")]
    text = _header_text((1,), descr)
    assert len(text.encode("utf8")) > len(text)
    path = _archive(
        tmp_path, {"values.npy": _member((1,), descr, version=(3, 0), payload=bytes(4))}
    )
    with np.load(path, allow_pickle=False, max_header_size=len(text)) as data:
        _forbid_array_reads(monkeypatch, data)
        header = inspect_npz_headers(data, ["values"])["values"]
        assert header.dtype.names == (field_name,) and header.nbytes == 4


def test_missing_fields_are_reported_without_array_access(tmp_path, monkeypatch):
    path = _archive(tmp_path, {"values.npy": _member((2,), payload=bytes(8))})
    with np.load(path, allow_pickle=False) as data:
        _forbid_array_reads(monkeypatch, data)
        with pytest.raises(ValueError, match="Missing NPZ field: absent"):
            inspect_npz_headers(data, ["absent"])
