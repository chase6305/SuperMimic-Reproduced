"""Small atomic, bounded run artifacts shared by training and deployment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_file(
    source: Path | str, destination: Path | str, *, expected_sha256: str | None = None
) -> str:
    """Publish a verified, read-only copy of one opened input inode.

    Training may atomically replace the source pathname while this function is
    copying it. Both reads stay on the original descriptor, so the returned
    digest always identifies the copied bytes. An in-place rewrite is rejected.
    The destination appears only after verification; a failed copy leaves any
    existing destination intact. Memory use is bounded independently of input size.
    """
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as reader:
            initial = os.fstat(reader.fileno())
            fd, name = tempfile.mkstemp(
                prefix="." + destination.name, suffix=".tmp", dir=destination.parent
            )
            temporary = Path(name)
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                writer.flush()
                os.fsync(writer.fileno())
            copied_digest = digest.hexdigest()
            reader.seek(0)
            verified = hashlib.file_digest(reader, "sha256").hexdigest()
            final = os.fstat(reader.fileno())
            if (
                verified != copied_digest
                or initial.st_size != final.st_size
                or initial.st_mtime_ns != final.st_mtime_ns
            ):
                raise RuntimeError(f"Input changed in place during snapshot: {source}")
        if expected_sha256 is not None and copied_digest != expected_sha256:
            raise ValueError(f"Input differs from the expected fingerprint: {source}")
        if sha256(temporary) != copied_digest:
            raise RuntimeError(f"Snapshot verification failed: {source}")
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        return copied_digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path | str, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
