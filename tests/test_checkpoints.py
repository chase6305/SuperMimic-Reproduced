"""Checkpoint provenance and CPU-only actor restoration share one read boundary."""

import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from supermimic.artifacts import sha256
from supermimic.checkpoints import (
    policy_config_from_checkpoint,
    policy_from_checkpoint,
    read_actor_checkpoint,
    read_checkpoint,
    write_checkpoint,
)
from supermimic.policy.nets import Policy, PolicyCfg


def test_checkpoint_evaluation_import_does_not_load_simulation_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class BlockSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mjlab', 'mujoco', 'warp'} or (
            fullname == 'supermimic.envs.tracking_env_cfg'
        ):
            raise AssertionError('Unexpected simulation dependency: ' + fullname)
sys.meta_path.insert(0, BlockSimulation())
from supermimic.eval import benchmark
from supermimic.eval.benchmark import load_policy_from_checkpoint
assert benchmark.load_policy_from_checkpoint is load_policy_from_checkpoint
assert 'supermimic.envs.tracking_env_cfg' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        check=True,
        timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


@pytest.mark.parametrize("mmap", [False, True, "auto"])
def test_read_keeps_original_inode_and_tensors_after_descriptor_closes(tmp_path, monkeypatch, mmap):
    checkpoint = tmp_path / "latest.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save({"iteration": 12, "value": torch.arange(4)}, checkpoint)
    torch.save({"iteration": 13, "value": torch.arange(4) + 10}, replacement)
    expected_hash = sha256(checkpoint)
    original_load = torch.load
    descriptors = []

    def publish_before_loading(source, **kwargs):
        assert kwargs["map_location"] == "cpu"
        assert kwargs["weights_only"] is False
        assert kwargs["mmap"] is (mmap is not False)
        descriptors.append(int(source.rsplit("/", 1)[-1]) if kwargs["mmap"] else source.fileno())
        replacement.replace(checkpoint)
        return original_load(source, **kwargs)

    monkeypatch.setattr(torch, "load", publish_before_loading)
    loaded = read_checkpoint(checkpoint, mmap=mmap, fingerprint=True)
    assert loaded.sha256 == expected_hash != sha256(checkpoint)
    assert loaded.data["iteration"] == 12
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    # mmap storages retain their mapping after the descriptor has been closed.
    torch.testing.assert_close(loaded.data["value"] + 1, torch.arange(4) + 1)
    assert loaded.data["value"].device.type == "cpu"


@pytest.mark.parametrize("mmap", [False, True, "auto"])
def test_fingerprint_detects_same_size_in_place_change(tmp_path, monkeypatch, mmap):
    checkpoint = tmp_path / "source.pt"
    torch.save({"value": torch.arange(4)}, checkpoint)
    original_load = torch.load

    def mutate_after_loading(source, **kwargs):
        loaded = original_load(source, **kwargs)
        stat = checkpoint.stat()
        with checkpoint.open("r+b") as destination:
            destination.seek(-1, os.SEEK_END)
            value = destination.read(1)
            destination.seek(-1, os.SEEK_END)
            destination.write(bytes([value[0] ^ 1]))
        os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return loaded

    monkeypatch.setattr(torch, "load", mutate_after_loading)
    with pytest.raises(RuntimeError, match="changed in place"):
        read_checkpoint(checkpoint, fingerprint=True, mmap=mmap)


def test_input_budget_is_checked_before_deserialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"larger than budget")

    def unexpected_load(*args, **kwargs):
        raise AssertionError("oversized checkpoint must not be deserialized")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="input budget"):
        read_checkpoint(checkpoint, max_bytes=1)


@pytest.mark.parametrize("mmap", [False, "auto"])
def test_reader_keeps_legacy_checkpoint_support(tmp_path, mmap):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"value": torch.arange(4)}, checkpoint, _use_new_zipfile_serialization=False)
    loaded = read_checkpoint(checkpoint, mmap=mmap)
    assert loaded.sha256 is None
    torch.testing.assert_close(loaded.data["value"], torch.arange(4))


def _actor_checkpoint():
    cfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,), history=2)
    actor = Policy(cfg)
    # JSON config snapshots contain lists, whereas dataclass defaults use tuples.
    config = json.loads(
        json.dumps({"policy": asdict(cfg), "env": {"future_frames": 10, "anchor_obs": False}})
    )
    return actor, {"config": config, "policy": actor.state_dict()}


def test_actor_restore_preserves_json_config_and_output():
    actor, checkpoint = _actor_checkpoint()
    original_config = json.dumps(checkpoint["config"])
    restored = policy_from_checkpoint(checkpoint)
    assert not restored.training
    assert restored.cfg.encoder_hidden == (8,)
    assert json.dumps(checkpoint["config"]) == original_config
    assert all(value.device.type == "cpu" for value in restored.state_dict().values())
    observation, goal = torch.randn(2, 186), torch.randn(2, 680)
    torch.testing.assert_close(restored(observation, goal), actor(observation, goal))


def test_actor_reader_owns_weights_and_metadata_after_source_is_rewritten(tmp_path):
    actor, checkpoint = _actor_checkpoint()
    checkpoint["anchor_adaptation"] = {"probe": torch.arange(3.0)}
    checkpoint["wm_opt"] = {"unused": torch.arange(4096)}

    def archive(data):
        stream = io.BytesIO()
        torch.save(data, stream)
        return stream.getvalue()

    source = tmp_path / "actor.pt"
    original = archive(checkpoint)
    source.write_bytes(original)
    restored = read_actor_checkpoint(source, fingerprint=True)
    expected_hash = sha256(source)
    replacement = deepcopy(checkpoint)
    replacement["policy"]["decoder.mlp.net.2.bias"].fill_(7)
    replacement["anchor_adaptation"]["probe"].add_(10)
    changed = archive(replacement)
    assert len(changed) == len(original)
    source.write_bytes(changed)  # The returned actor must no longer depend on this inode's pages.
    assert restored.sha256 == expected_hash != sha256(source)
    assert "wm_opt" not in restored.metadata
    torch.testing.assert_close(restored.metadata["anchor_adaptation"]["probe"], torch.arange(3.0))
    obs, goal = torch.randn(2, 186), torch.randn(2, 680)
    torch.testing.assert_close(restored.actor(obs, goal), actor(obs, goal), atol=0, rtol=0)


def test_actor_copy_is_inside_final_checkpoint_fingerprint_check(tmp_path, monkeypatch):
    import supermimic.checkpoints as module

    _, checkpoint = _actor_checkpoint()
    source = tmp_path / "actor.pt"
    torch.save(checkpoint, source)
    original_restore = module.policy_from_checkpoint

    def restore_then_change(data):
        actor = original_restore(data)
        stat = source.stat()
        with source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            byte = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([byte[0] ^ 1]))
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return actor

    monkeypatch.setattr(module, "policy_from_checkpoint", restore_then_change)
    with pytest.raises(RuntimeError, match="changed in place"):
        read_actor_checkpoint(source, fingerprint=True)


@pytest.mark.parametrize("field,value", [("future_frames", 9), ("anchor_obs", True)])
def test_actor_restore_rejects_inconsistent_goal_configuration(field, value):
    _, checkpoint = _actor_checkpoint()
    checkpoint["config"]["env"][field] = value
    with pytest.raises(ValueError, match="policy/goal"):
        policy_config_from_checkpoint(checkpoint)


@pytest.mark.parametrize("key", ["encoders.motion.scale", "decoder.mlp.net.0.weight"])
def test_actor_restore_rejects_nonfinite_parameters_and_buffers(key):
    _, checkpoint = _actor_checkpoint()
    checkpoint["policy"][key].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite actor"):
        policy_from_checkpoint(checkpoint)


def test_evaluation_loads_checkpoint_on_cpu_and_moves_only_actor(tmp_path, monkeypatch):
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    _, checkpoint = _actor_checkpoint()
    checkpoint["wm_opt"] = {"unused_optimizer_tensor": torch.arange(8)}
    path = tmp_path / "training.pt"
    torch.save(checkpoint, path)
    original_load = torch.load
    locations = []
    destinations = []

    def load_on_cpu(source, **kwargs):
        locations.append(kwargs["map_location"])
        return original_load(source, **kwargs)

    def record_actor_move(self, destination):
        destinations.append(destination)
        return self

    monkeypatch.setattr(torch, "load", load_on_cpu)
    monkeypatch.setattr(Policy, "to", record_actor_move)
    actor, config = load_policy_from_checkpoint(Path(path), "cuda:7")
    assert locations == ["cpu"]
    assert destinations == ["cuda:7"]
    assert actor.cfg.history == 2 and not actor.training
    assert config == checkpoint["config"]


def test_evaluation_keeps_legacy_checkpoint_serialization_support(tmp_path):
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    original_actor, checkpoint = _actor_checkpoint()
    path = tmp_path / "legacy-policy.pt"
    torch.save(checkpoint, path, _use_new_zipfile_serialization=False)

    restored, config = load_policy_from_checkpoint(path, "cpu")

    observation, goal = torch.randn(2, 186), torch.randn(2, 680)
    torch.testing.assert_close(restored(observation, goal), original_actor(observation, goal))
    assert config == checkpoint["config"]
    assert all(value.device.type == "cpu" for value in restored.state_dict().values())


def test_checkpoint_writer_preserves_previous_mapped_publication(tmp_path):
    path = tmp_path / "latest.pt"
    write_checkpoint(path, {"iteration": 1, "value": torch.arange(4)})
    mapped = read_checkpoint(path, mmap=True)
    with pytest.raises(FileExistsError):
        write_checkpoint(path, {"iteration": 2})
    assert read_checkpoint(path).data["iteration"] == 1
    write_checkpoint(path, {"iteration": 3, "value": torch.arange(4) + 10}, overwrite=True)
    assert read_checkpoint(path).data["iteration"] == 3
    torch.testing.assert_close(mapped.data["value"], torch.arange(4))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_checkpoint_serialization_removes_partial_file(tmp_path, monkeypatch, overwrite):
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous complete publication")

    def partial_write(data, stream):
        stream.write(b"incomplete new checkpoint")
        raise OSError("disk write failed")

    monkeypatch.setattr(torch, "save", partial_write)
    with pytest.raises(OSError, match="disk write failed"):
        write_checkpoint(path, {"iteration": 2}, overwrite=overwrite)
    assert path.read_bytes() == b"previous complete publication"
    assert list(tmp_path.iterdir()) == [path]
