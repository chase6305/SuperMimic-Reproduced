"""Input snapshots retain one published inode and never expose a partial copy."""

import hashlib
import os
from pathlib import Path

import pytest

from supermimic.artifacts import snapshot_file


def test_snapshot_is_verified_read_only_and_does_not_change_source_permissions(tmp_path):
    source = tmp_path / "source.pt"
    payload = b"published checkpoint" * 70000
    source.write_bytes(payload)
    source.chmod(0o640)
    destination = tmp_path / "inputs/checkpoint.pt"

    digest = snapshot_file(source, destination)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o222 == 0
    assert source.stat().st_mode & 0o777 == 0o640
    assert list(destination.parent.iterdir()) == [destination]


def _after_first_source_read(monkeypatch, source, action):
    original_open = Path.open
    called = False

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            nonlocal called
            result = self.stream.read(size)
            if not called:
                called = True
                action()
            return result

    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == source and args and args[0] == "rb":
            return Reader(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_path)


def test_snapshot_keeps_original_inode_when_latest_path_is_atomically_replaced(
    tmp_path, monkeypatch
):
    source, replacement = tmp_path / "latest.pt", tmp_path / "next.pt"
    source.write_bytes(b"original policy")
    replacement.write_bytes(b"next policy")
    _after_first_source_read(monkeypatch, source, lambda: os.replace(replacement, source))
    destination = tmp_path / "snapshot.pt"

    digest = snapshot_file(source, destination)

    assert destination.read_bytes() == b"original policy"
    assert digest == hashlib.sha256(b"original policy").hexdigest()
    assert source.read_bytes() == b"next policy"


def test_in_place_mutation_aborts_snapshot_and_preserves_existing_destination(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "source.pt", tmp_path / "snapshot.pt"
    source.write_bytes(b"original policy")
    destination.write_bytes(b"previous verified snapshot")
    _after_first_source_read(monkeypatch, source, lambda: source.write_bytes(b"rewritten policy"))

    with pytest.raises(RuntimeError, match="changed in place"):
        snapshot_file(source, destination)

    assert destination.read_bytes() == b"previous verified snapshot"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["snapshot.pt", "source.pt"]


def test_manifest_mismatch_does_not_publish_untrusted_input(tmp_path):
    source, destination = tmp_path / "source.npz", tmp_path / "snapshot.npz"
    source.write_bytes(b"unexpected motion")

    with pytest.raises(ValueError, match="expected fingerprint"):
        snapshot_file(source, destination, expected_sha256=hashlib.sha256(b"expected").hexdigest())

    assert list(tmp_path.iterdir()) == [source]


def test_publish_failure_leaves_existing_destination_intact_and_removes_temporary(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "source.pt", tmp_path / "snapshot.pt"
    source.write_bytes(b"new policy")
    destination.write_bytes(b"previous policy")

    def denied(*args):
        raise PermissionError("publication denied")

    monkeypatch.setattr("supermimic.artifacts.os.replace", denied)
    with pytest.raises(PermissionError, match="publication denied"):
        snapshot_file(source, destination)

    assert destination.read_bytes() == b"previous policy"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["snapshot.pt", "source.pt"]
