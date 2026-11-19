"""Read trusted local checkpoints on CPU without losing their published inode.

Training publishes checkpoints by atomic replacement. Holding one descriptor keeps
the loaded tensors and optional provenance digest tied to the same publication.
Memory mapping is for these immutable, atomically published files; callers must
not modify their contents in place while mapped tensors remain in use.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


def validate_optimizer_resume(checkpoint: Mapping[str, Any]) -> None:
    """Reject explicitly stale or unverified optimizer state before strict restore.

    Some experimental artifacts materialize new weights while retaining their
    source Adam history for forensics. Only the latest declarations apply;
    historical parent provenance must not invalidate a fresh optimizer chain.
    Legacy archives with the standard optimizer layout retain their behavior.
    """
    invalid = []
    for block, field, label in (
        ("native_policy_improvement", "policy_optimizer_resume_valid", "policy"),
        ("native_policy_improvement", "critic_optimizer_resume_valid", "critic"),
        ("native_dynamics_calibration", "optimizer_resume_valid", "world model"),
    ):
        metadata = checkpoint.get(block)
        if not isinstance(metadata, Mapping) or field not in metadata:
            continue
        value = metadata[field]
        if type(value) is not bool:
            raise ValueError(f"Checkpoint {block}.{field} must be boolean")
        if not value:
            invalid.append(label)
    metadata = checkpoint.get("native_dynamics_calibration")
    if isinstance(metadata, Mapping) and "optimizer_correspondence_not_proved" in metadata:
        value = metadata["optimizer_correspondence_not_proved"]
        if type(value) is not bool:
            raise ValueError("Checkpoint optimizer_correspondence_not_proved must be boolean")
        if value and "world model" not in invalid:
            invalid.append("world model")
    if invalid:
        raise ValueError(
            "Checkpoint declares stale or unverified "
            + ", ".join(invalid)
            + " optimizer state; use --transfer for weights with fresh optimizers"
        )
    # Standard trainers construct one Adam group each. A calibrated checkpoint
    # may have valid moments but a different grouping, which Adam cannot load.
    # Diagnose that before allocating the simulator or changing any weights.
    for key, label in (
        ("wm_opt", "world model"),
        ("policy_opt", "policy"),
        ("critic_opt", "critic"),
    ):
        state = checkpoint.get(key)
        if not isinstance(state, Mapping) or "param_groups" not in state:
            continue
        groups = state["param_groups"]
        if not isinstance(groups, (list, tuple)) or len(groups) != 1:
            raise ValueError(
                f"Checkpoint {label} optimizer parameter groups differ from the single group "
                "required by the standard trainer; use --transfer for weights with fresh optimizers"
            )


@dataclass(frozen=True)
class ActorCheckpointRead:
    actor: Policy
    metadata: dict[str, Any]
    sha256: str | None


def read_actor_checkpoint(path: Path | str, *, fingerprint: bool = False) -> ActorCheckpointRead:
    """Copy the actor and its metadata while the source inode remains checked.

    Modern checkpoints can map unused optimizer/world-model storage. Returned
    weights and metadata own their storage, so subsequent source changes cannot
    alter the actor or its provenance after the reader's final verification.
    """

    def restore(data):
        if not isinstance(data, dict):
            raise ValueError("Expected a checkpoint dictionary")
        metadata = {
            key: deepcopy(data[key])
            for key in (
                "config",
                "iteration",
                "control_contract",
                "motion_files",
                "anchor_adaptation",
                "transfer_provenance",
            )
            if key in data
        }
        actor = policy_from_checkpoint({"config": metadata["config"], "policy": data["policy"]})
        return actor, metadata

    loaded = read_torch_archive(path, mmap="auto", fingerprint=fingerprint, consume=restore)
    actor, metadata = loaded.data
    return ActorCheckpointRead(actor, metadata, loaded.sha256)


def read_checkpoint(
    path: Path | str,
    *,
    mmap: bool | Literal["auto"] = False,
    fingerprint: bool = False,
    max_bytes: int | None = None,
) -> CheckpointRead:
    """Load CPU storage, optionally mapping it or computing a matching SHA-256.

    PyTorch accepts mmap only for a pathname. On Linux, /proc/self/fd exposes the
    open descriptor as a pathname, so an atomic replacement of ``path`` cannot
    redirect the mapping. The mapping outlives the descriptor after return.
    Explicit ``True`` has no eager fallback: callers choose the Linux mapping
    requirement to avoid unexpectedly materializing optimizer tensors. ``auto``
    maps modern ZIP checkpoints when descriptor paths are available and keeps
    the eager reader for legacy archives and other platforms.

    Pickle loading remains enabled for existing training checkpoints: inputs must
    be trusted, as at the training, migration and export entry points.
    """
    loaded = read_torch_archive(path, mmap=mmap, fingerprint=fingerprint, max_bytes=max_bytes)
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
    actor.load_state_dict(checkpoint["policy"], strict=True, assign=False)
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite actor parameters or buffers")
    return actor.eval()
