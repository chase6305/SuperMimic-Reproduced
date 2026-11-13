"""Generic artifact reads select a format on the same CPU input publication."""

import os
from pathlib import Path

import pytest
import torch

from supermimic.artifacts import sha256
from supermimic.checkpoints import read_checkpoint
from supermimic.torch_io import read_torch_archive, write_torch_archive


@pytest.mark.parametrize("modern", [True, False], ids=["zip-to-legacy", "legacy-to-zip"])
def test_auto_format_selection_and_digest_remain_on_original_inode(tmp_path, monkeypatch, modern):
    path = tmp_path / "current.pt"
    replacement = tmp_path / "replacement.pt"
    expected = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    torch.save(("original", expected), path, _use_new_zipfile_serialization=modern)
    torch.save(("replacement", expected + 100), replacement, _use_new_zipfile_serialization=not modern)
    expected_digest = sha256(path)
    original_load = torch.load
    mappings = []

    def publish_after_format_detection(source, **kwargs):
        assert kwargs["map_location"] == "cpu"
        assert kwargs["weights_only"] is False
        mappings.append(kwargs["mmap"])
        if kwargs["mmap"]:
            assert isinstance(source, str) and source.startswith("/proc/self/fd/")
        else:
            assert hasattr(source, "fileno") and source.tell() == 0
        # Detection has already selected ZIP/mmap or the legacy eager loader.
        # Reopening this pathname would now choose a different serialization.
        replacement.replace(path)
        return original_load(source, **kwargs)

    monkeypatch.setattr(torch, "load", publish_after_format_detection)
    loaded = read_torch_archive(path, mmap="auto", fingerprint=True)

    assert mappings == [modern and Path("/proc/self/fd").is_dir()]
    assert loaded.sha256 == expected_digest != sha256(path)
    assert loaded.data[0] == "original"
    assert loaded.data[1].device.type == "cpu"
    torch.testing.assert_close(loaded.data[1], expected)


def test_auto_uses_eager_when_descriptor_path_is_unavailable_but_explicit_mmap_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "tensor.pt"
    tensor = torch.arange(5, dtype=torch.float32)
    torch.save(tensor, path)
    original_exists = Path.exists
    original_load = torch.load
    loads = []

    def exists_without_proc(self):
        return False if str(self).startswith("/proc/self/fd/") else original_exists(self)

    def eager_load(source, **kwargs):
        assert hasattr(source, "fileno")
        assert kwargs["map_location"] == "cpu" and kwargs["mmap"] is False
        loads.append(source.fileno())
        return original_load(source, **kwargs)

    monkeypatch.setattr(Path, "exists", exists_without_proc)
    monkeypatch.setattr(torch, "load", eager_load)

    loaded = read_torch_archive(path, mmap="auto", fingerprint=True)
    torch.testing.assert_close(loaded.data, tensor)
    assert loaded.data.device.type == "cpu" and loaded.sha256 == sha256(path)
    assert len(loads) == 1

    with pytest.raises(RuntimeError, match="/proc/self/fd"):
        read_torch_archive(path, mmap=True)
    assert len(loads) == 1  # Explicit mmap did not silently call the eager reader.


@pytest.mark.parametrize("kind", ["tensor", "tuple"])
def test_generic_non_dict_payloads_round_trip_without_weakening_checkpoint_schema(tmp_path, kind):
    path = tmp_path / "artifact.pt"
    tensor = torch.tensor([1.25, -2.5, 3.75], dtype=torch.float64)
    payload = tensor if kind == "tensor" else ("reference", [tensor, {"frames": 3}])
    write_torch_archive(path, payload)

    loaded = read_torch_archive(path, mmap="auto")
    actual = loaded.data if kind == "tensor" else loaded.data[1][0]
    torch.testing.assert_close(actual, tensor)
    assert actual.device.type == "cpu" and loaded.sha256 is None
    if kind == "tuple":
        assert loaded.data[0] == "reference" and loaded.data[1][1] == {"frames": 3}
    with pytest.raises(ValueError, match="checkpoint dictionary"):
        read_checkpoint(path)


def test_failed_fsync_preserves_publication_and_closes_temporary_archive(tmp_path, monkeypatch):
    path = tmp_path / "artifact.pt"
    write_torch_archive(path, ["original", torch.arange(3)])
    original_digest = sha256(path)
    descriptors = []

    def reject_fsync(fd):
        descriptors.append(fd)
        raise OSError("artifact fsync failed")

    monkeypatch.setattr("supermimic.torch_io.os.fsync", reject_fsync)
    with pytest.raises(OSError, match="artifact fsync failed"):
        write_torch_archive(path, ["replacement", torch.arange(4)], overwrite=True)

    assert sha256(path) == original_digest
    assert list(tmp_path.iterdir()) == [path]
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
