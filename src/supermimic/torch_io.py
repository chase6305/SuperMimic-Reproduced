"""Atomic storage for trusted local Torch artifacts, independent of their schema.

Callers validate checkpoint or motion semantics after CPU deserialization. Mapped
storage requires immutable, atomically published inputs while tensors are in use.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class TorchArchive:
    data: Any
    sha256: str | None


def read_torch_archive(
    path: Path | str,
    *,
    mmap: bool | Literal["auto"] = False,
    fingerprint: bool = False,
    max_bytes: int | None = None,
) -> TorchArchive:
    """Read CPU storage and an optional matching digest from one opened inode.

    Explicit mmap requires Linux /proc/self/fd, with no eager fallback. ``auto``
    maps modern ZIP serialization when that descriptor path is available and
    otherwise keeps the eager reader's legacy-format/platform support. Automatic
    format selection and deserialization inspect the same descriptor.

    Pickle remains enabled for existing local artifacts; inputs must be trusted.
    """
    # torch.load performs small reads/seeks; buffered bytes could hide an
    # in-place rewrite from the second digest after the reader is rewound.
    with Path(path).open("rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"Torch archive exceeds the {max_bytes} byte input budget")
        digest = hashlib.file_digest(stream, "sha256").hexdigest() if fingerprint else None
        stream.seek(0)
        descriptor_path = f"/proc/self/fd/{stream.fileno()}"
        if mmap == "auto":
            mapping = Path(descriptor_path).exists() and stream.read(4) == b"PK\x03\x04"
            stream.seek(0)
        else:
            mapping = mmap
        source = stream
        if mapping:
            if not Path(descriptor_path).exists():
                raise RuntimeError("Torch archive mmap requires Linux /proc/self/fd")
            source = descriptor_path
        data = torch.load(source, map_location="cpu", weights_only=False, mmap=mapping)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Torch archive changed in place during loading")
        if fingerprint:
            stream.seek(0)
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise RuntimeError("Torch archive changed in place during loading")
    return TorchArchive(data, digest)


def write_torch_archive(
    path: Path | str, data: Any, *, overwrite: bool = False
) -> None:
    """Flush a complete archive before exclusive or replacing publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="." + path.name + "-", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
