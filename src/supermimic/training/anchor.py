"""Warm-start the optional relative-anchor interface without discarding learned state.

This is an explicit extension to the proprioception-only baseline, not an
equivalent reproduction of its observation interface. Only trusted checkpoints
are accepted, just like training/export.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from supermimic.artifacts import sha256, write_json
from supermimic.checkpoints import policy_from_checkpoint, read_checkpoint, write_checkpoint
from supermimic.interfaces import ANCHOR_OBS_DIM, goal_dim, observation_dim
from supermimic.policy.nets import Critic, CriticCfg, Policy


def _insert_columns(value: torch.Tensor, offset: int) -> torch.Tensor:
    return torch.cat(
        [value[:, :offset], value.new_zeros(value.shape[0], ANCHOR_OBS_DIM), value[:, offset:]], 1
    )


def _expand_optimizer(optimizer: dict, model: torch.nn.Module, name: str, offset: int) -> None:
    groups = optimizer["param_groups"]
    # The trainer creates a single group with model.parameters() in registration order.
    params = list(model.named_parameters())
    if len(groups) != 1 or len(groups[0]["params"]) != len(params):
        raise ValueError("Unsupported optimizer parameter layout")
    index = [key for key, _ in params].index(name)
    state = optimizer["state"].get(groups[0]["params"][index], {})
    original = params[index][1]
    for key, value in state.items():
        if key == "step":
            continue
        if key not in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") or value.shape != original.shape:
            raise ValueError("Unsupported optimizer state for expanded anchor weights")
        state[key] = _insert_columns(value, offset)


def enable_anchor_feedback(checkpoint: Path, output: Path) -> dict:
    """Add nine initially ignored inputs and preserve actor/critic Adam moments."""
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    loaded = read_checkpoint(checkpoint, fingerprint=True, max_bytes=512 * 1024**2)
    ck = loaded.data
    cfg = ck["config"]
    if cfg["env"].get("anchor_obs", False):
        raise ValueError("Checkpoint already has anchor feedback")
    actor = policy_from_checkpoint(ck)
    pcfg = actor.cfg
    old_dim = goal_dim(cfg["env"]["future_frames"])
    critic = Critic(CriticCfg(**cfg["critic"]), pcfg=pcfg).eval()
    critic.load_state_dict(ck["critic"], strict=True)
    if not all(torch.isfinite(value).all() for value in critic.state_dict().values()):
        raise ValueError("Checkpoint has non-finite actor/critic weights")
    actor_key, critic_key = "encoders.motion.mlp.net.0.weight", "mlp.net.0.weight"
    critic_offset = 71 + old_dim
    _expand_optimizer(ck["policy_opt"], actor, actor_key, old_dim)
    _expand_optimizer(ck["critic_opt"], critic, critic_key, critic_offset)
    ck["policy"][actor_key] = _insert_columns(ck["policy"][actor_key], old_dim)
    ck["critic"][critic_key] = _insert_columns(ck["critic"][critic_key], critic_offset)
    for state, key in ((ck["policy"], "encoders.motion.scale"), (ck["critic"], "goal_scale")):
        state[key] = torch.cat([state[key], state[key].new_ones(ANCHOR_OBS_DIM)])
    new_cfg = replace(pcfg, goal_dim=old_dim + ANCHOR_OBS_DIM)
    adapted_actor = Policy(new_cfg).eval()
    adapted_critic = Critic(CriticCfg(**cfg["critic"]), pcfg=new_cfg).eval()
    adapted_actor.load_state_dict(ck["policy"], strict=True)
    adapted_critic.load_state_dict(ck["critic"], strict=True)
    generator = torch.Generator().manual_seed(617)
    errors = {"actor": 0.0, "critic": 0.0}
    with torch.inference_mode():
        for n in (1, 4):
            obs = torch.randn(n, observation_dim(pcfg.history), generator=generator)
            goal = torch.randn(n, old_dim, generator=generator)
            anchor = torch.randn(n, ANCHOR_OBS_DIM, generator=generator) * 3
            extended = torch.cat([goal, anchor], -1)
            feat = torch.randn(n, 71, generator=generator)
            xi = torch.randn(n, 33, generator=generator)
            error = torch.randn(n, 4, generator=generator)
            for name, before, after in (
                ("actor", actor(obs, goal), adapted_actor(obs, extended)),
                (
                    "critic",
                    critic(feat, goal, xi, error),
                    adapted_critic(feat, extended, xi, error),
                ),
            ):
                torch.testing.assert_close(after, before, atol=2e-6, rtol=1e-5)
                errors[name] = max(errors[name], float((after - before).abs().max()))
    cfg["policy"]["goal_dim"] = old_dim + ANCHOR_OBS_DIM
    cfg["env"]["anchor_obs"] = True
    cfg["policy_trainer"]["anchor_obs"] = True
    provenance = {
        "source_checkpoint_sha256": loaded.sha256,
        "source_iteration": ck["iteration"],
        "old_goal_dim": old_dim,
        "new_goal_dim": old_dim + ANCHOR_OBS_DIM,
        "interface": "reference torso relative to actual torso: position(3), rotation columns(6)",
        "initialization": "zero extra weight columns; preserve original weights and Adam moments",
        "parity_max_abs_error": errors,
        "scope": "Observation-interface extension; requires MuJoCo pose feedback, no hardware validation",
    }
    ck["anchor_adaptation"] = provenance
    write_checkpoint(output, ck)
    provenance = {**provenance, "checkpoint_sha256": sha256(output)}
    write_json(output.with_suffix(output.suffix + ".adaptation.json"), provenance)
    return provenance
