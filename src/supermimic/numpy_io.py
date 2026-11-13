"""Inspect NPZ array headers before a consumer allocates their declared payloads.

Archive ownership and total ZIP budgets belong to the caller. Header inspection
uses the same member mapping as NpzFile, leaves the archive open, and never loads
an array. Domain code still validates array shapes and values before consuming
them, including zero-width string arrays whose byte size alone is not a bound
on the number of Python objects produced by ``tolist``.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.lib import format
from numpy.lib.npyio import NpzFile

try:
    from numpy.lib._format_impl import _read_array_header
except ImportError:  # NumPy versions before the format implementation was split.
    from numpy.lib.format import _read_array_header

_HEADER_FORMATS = {
    (1, 0): ("<H", 1),  # Latin-1: at most one byte per decoded character.
    (2, 0): ("<I", 1),
    (3, 0): ("<I", 4),  # UTF-8: up to four bytes per decoded character.
}


@dataclass(frozen=True)
class ArrayHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    nbytes: int


def _member_name(data: NpzFile, field: str) -> str:
    members = data._files
    if isinstance(members, Mapping):
        try:
            return members[field]
        except KeyError:
            raise ValueError(f"Missing NPZ field: {field}") from None
    # Older NpzFile keeps raw names as a sequence. Its getitem checks an exact
    # raw member before trying the public name with an appended .npy suffix.
    if field in members:
        return field
    if field in data.files:
        return field + ".npy"
    raise ValueError(f"Missing NPZ field: {field}")


def inspect_npz_headers(data: NpzFile, fields: Iterable[str]) -> dict[str, ArrayHeader]:
    """Read safe NPY metadata for selected fields without materializing arrays.

    Explicit object arrays, negative dimensions, truncated payloads and headers
    beyond the NpzFile's own header limit are rejected. Extra payload tail bytes
    and valid zero-width string dtypes retain NumPy's existing semantics.
    """
    headers = {}
    for field in fields:
        member_name = _member_name(data, field)
        info = data.zip.getinfo(member_name)
        with data.zip.open(info) as member:
            version = format.read_magic(member)
            try:
                length_format, max_bytes_per_character = _HEADER_FORMATS[version]
            except KeyError:
                raise ValueError(f"Unsupported NPY version {version} for NPZ field {field}") from None
            header_start = member.tell()
            length_size = struct.calcsize(length_format)
            length_bytes = member.read(length_size)
            if len(length_bytes) != length_size:
                raise ValueError(f"Truncated NPY header length for NPZ field {field}")
            header_length = struct.unpack(length_format, length_bytes)[0]
            if header_length > info.file_size - member.tell():
                raise ValueError(f"NPY header exceeds member size for NPZ field {field}")
            if header_length > data.max_header_size * max_bytes_per_character:
                raise ValueError(f"NPY header exceeds header budget for NPZ field {field}")
            # NumPy handles version-specific decoding, Python-2 header filtering,
            # dtype descriptors and the exact decoded-character header limit.
            member.seek(header_start)
            shape, fortran_order, dtype = _read_array_header(
                member, version, max_header_size=data.max_header_size
            )
            if dtype.hasobject:
                raise ValueError(f"Object dtype is unsupported for NPZ field {field}")
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"Negative shape is unsupported for NPZ field {field}")
            # Python integers avoid the int64 overflow in array-size products.
            nbytes = math.prod(shape) * dtype.itemsize
            if nbytes > info.file_size - member.tell():
                raise ValueError(f"NPY payload exceeds member size for NPZ field {field}")
            headers[field] = ArrayHeader(shape, dtype, fortran_order, nbytes)
    return headers
