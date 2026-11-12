"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from supermimic.config import CriticCfg as CriticCfg
from supermimic.config import PolicyCfg as PolicyCfg
from supermimic.interfaces import ACT_DIM as ACT_DIM
from supermimic.interfaces import GOAL_DIM as GOAL_DIM
from supermimic.interfaces import GOAL_FEAT_DIM, GOAL_TERMS, OBS_TERMS, observation_dim
from supermimic.world_model.dynamics import MLP

LATENT_DIM = 32


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (_name, d), s in zip(OBS_TERMS, cfg.obs_term_scale, strict=True):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = tuple(size for _, size in GOAL_TERMS)
    per_frame = torch.cat(
        [torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale, strict=True)]
    )
    k, extra = divmod(goal_dim, GOAL_FEAT_DIM)
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int = observation_dim(10)):
        super().__init__()
        self.mlp = MLP(
            cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, last_gain=cfg.decoder_last_gain
        )
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        if cfg.action_clip is not None and (
            not math.isfinite(cfg.action_clip) or cfg.action_clip <= 0
        ):
            raise ValueError("Action clipping must be finite and positive")
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg, obs_dim=observation_dim(cfg.history))

    def forward(
        self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion"
    ) -> torch.Tensor:
        action = self.decoder(self.encoders[interface](goal), obs_hist)
        return (
            action
            if self.cfg.action_clip is None
            else action.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        )


class Critic(nn.Module):
    """V(canonical state 71, goal 680, xi 33, root error 4) -> scalar (one per interface)."""

    def __init__(
        self,
        cfg: CriticCfg,
        feat_dim: int = 71,
        goal_dim: int | None = None,
        xi_dim: int = 33,
        pcfg: PolicyCfg | None = None,
    ):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(
        self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(
            -1
        )
