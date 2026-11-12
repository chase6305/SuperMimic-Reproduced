"""Read trusted local checkpoints on CPU without losing their published inode.

Training publishes checkpoints by atomic replacement. Holding one descriptor keeps
the loaded tensors and optional provenance digest tied to the same publication.
Memory mapping is for these immutable, atomically published files; callers must
not modify their contents in place while mapped tensors remain in use.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from supermimic.interfaces import goal_dim

if TYPE_CHECKING:
    from supermimic.config import PolicyCfg
    from supermimic.policy.nets import Policy


@dataclass(frozen=True)
class CheckpointRead:
    data: dict[str, Any]
    sha256: str | None


def read_checkpoint(
    path: Path | str,
    *,
    mmap: bool = False,
    fingerprint: bool = False,
    max_bytes: int | None = None,
) -> CheckpointRead:
    """Load CPU storage, optionally mapping it or computing a matching SHA-256.

    PyTorch accepts mmap only for a pathname. On Linux, /proc/self/fd exposes the
    open descriptor as a pathname, so an atomic replacement of ``path`` cannot
    redirect the mapping. The mapping outlives the descriptor after return. There
    is no eager fallback: mmap callers explicitly choose the Linux file mapping
    requirement to avoid unexpectedly materializing optimizer tensors.

    Pickle loading remains enabled for existing training checkpoints: inputs must
    be trusted, as at the training, migration and export entry points.
    """
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"Checkpoint exceeds the {max_bytes} byte input budget")
        digest = hashlib.file_digest(stream, "sha256").hexdigest() if fingerprint else None
        stream.seek(0)
        source = stream
        if mmap:
            source = f"/proc/self/fd/{stream.fileno()}"
            if not Path(source).exists():
                raise RuntimeError("Checkpoint mmap requires Linux /proc/self/fd")
        data = torch.load(source, map_location="cpu", weights_only=False, mmap=mmap)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Checkpoint changed in place during loading")
        if fingerprint:
            stream.seek(0)
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise RuntimeError("Checkpoint changed in place during loading")
    if not isinstance(data, dict):
        raise ValueError("Expected a checkpoint dictionary")
    return CheckpointRead(data, digest)


def write_checkpoint(path: Path | str, data: dict[str, Any], *, overwrite: bool = False) -> None:
    """Publish a complete checkpoint, cleaning up partial serialization on failure.

    New exports/migrations use exclusive publication. Only the training writer
    explicitly replaces its latest or numbered publication. Neither mode mutates
    an inode that a reader may still have mapped.
    """
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


def policy_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> PolicyCfg:
    """Restore the existing policy configuration and validate its goal interface."""
    from supermimic.config import PolicyCfg

    cfg = checkpoint["config"]
    policy = cfg["policy"]
    pcfg = PolicyCfg(
        **{key: tuple(value) if isinstance(value, list) else value for key, value in policy.items()}
    )
    env = cfg["env"]
    if pcfg.goal_dim != goal_dim(env["future_frames"], env.get("anchor_obs", False)):
        raise ValueError("Checkpoint policy/goal configuration is inconsistent")
    if type(pcfg.history) is not int or pcfg.history < 1:
        raise ValueError("Checkpoint observation history must be a positive integer")
    return pcfg


def policy_from_checkpoint(checkpoint: Mapping[str, Any]) -> Policy:
    """Build the actor on CPU; optimizer and world-model storage stay on CPU."""
    from supermimic.policy.nets import Policy

    pcfg = policy_config_from_checkpoint(checkpoint)
    with torch.device("cpu"):
        actor = Policy(pcfg)
    actor.load_state_dict(checkpoint["policy"], strict=True)
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite actor parameters or buffers")
    return actor.eval()
