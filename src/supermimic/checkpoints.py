"""Read trusted local checkpoints on CPU without losing their published inode.

Training publishes checkpoints by atomic replacement. Holding one descriptor keeps
the loaded tensors and optional provenance digest tied to the same publication.
Memory mapping is for these immutable, atomically published files; callers must
not modify their contents in place while mapped tensors remain in use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from supermimic.interfaces import goal_dim
from supermimic.torch_io import read_torch_archive, write_torch_archive

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
    loaded = read_torch_archive(
        path, mmap=mmap, fingerprint=fingerprint, max_bytes=max_bytes
    )
    if not isinstance(loaded.data, dict):
        raise ValueError("Expected a checkpoint dictionary")
    return CheckpointRead(loaded.data, loaded.sha256)


def write_checkpoint(path: Path | str, data: dict[str, Any], *, overwrite: bool = False) -> None:
    """Publish a complete checkpoint, cleaning up partial serialization on failure.

    Migrated checkpoints use exclusive publication. Only the training writer
    explicitly replaces its latest or numbered publication. Neither mode mutates
    an inode that a reader may still have mapped.
    """
    write_torch_archive(path, data, overwrite=overwrite)


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
