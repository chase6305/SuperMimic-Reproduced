"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from supermimic.config import PolicyTrainerCfg as PolicyTrainerCfg
from supermimic.data.motion_library import MotionLibrary
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.imagination.objective import gae_targets, policy_objective, survival_weighted_mse
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.imagination.tracking import tracking_auxiliary
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


def configure_policy_update_scope(policy: Policy, scope: str) -> None:
    """Freeze actor parameters without detaching feedback derivatives or changing Adam groups."""
    if scope not in ("all", "decoder", "action-head"):
        raise ValueError("Policy update scope must be all, decoder or action-head")
    if scope == "all":
        selected = policy
    elif scope == "decoder":
        selected = policy.decoder
    else:
        selected = policy.decoder.mlp.net[-1]
        if not isinstance(selected, torch.nn.Linear):
            raise ValueError("Policy action head must be the decoder's final Linear layer")
    policy.requires_grad_(False)
    selected.requires_grad_(True)
    for parameter in policy.parameters():
        if not parameter.requires_grad:
            parameter.grad = None


@dataclass
class CriticBatch:
    features: torch.Tensor
    goals: torch.Tensor
    root_error: torch.Tensor
    context: torch.Tensor
    rewards: torch.Tensor
    termination: torch.Tensor
    survival: torch.Tensor


class PolicyTrainer:
    def __init__(
        self,
        wm: WorldModel,
        policy: Policy,
        critic: Critic,
        lib: MotionLibrary,
        cfg: PolicyTrainerCfg,
        *,
        support_model=None,
    ):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        if (
            not isinstance(cfg.gradient_accumulation_steps, int)
            or not 1 <= cfg.gradient_accumulation_steps <= 16
        ):
            raise ValueError("Policy gradient accumulation must be an integer within [1, 16]")
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                cfg.joint_tracking_weight,
                cfg.root_xy_tracking_weight,
                cfg.height_tracking_weight,
                cfg.orientation_tracking_weight,
                cfg.vertical_velocity_tracking_weight,
                cfg.angular_velocity_tracking_weight,
                cfg.foot_support_height_weight,
                cfg.foot_support_velocity_weight,
                cfg.effector_position_weight,
            )
        ):
            raise ValueError("Auxiliary tracking weights must be finite and nonnegative")
        if len(cfg.effector_position_weights) != 4 or any(
            not math.isfinite(v) or v <= 0 for v in cfg.effector_position_weights
        ):
            raise ValueError("Effector position weights must contain four finite positive values")
        if cfg.orientation_loss not in ("chordal", "geodesic"):
            raise ValueError("Unknown orientation loss")
        if not 0 <= cfg.reset_start_fraction <= 1:
            raise ValueError("Reset start fraction must be within [0, 1]")
        if len(cfg.joint_tracking_weights) != 29 or any(
            not math.isfinite(v) or v <= 0 for v in cfg.joint_tracking_weights
        ):
            raise ValueError("Joint tracking weights must have 29 finite positive values")
        configure_policy_update_scope(policy, cfg.update_scope)
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import ImagStep

        self.step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile)
        self.foot_support = None
        if (
            cfg.foot_support_height_weight
            or cfg.foot_support_velocity_weight
            or cfg.effector_position_weight
        ):
            from supermimic.imagination.foot_support import FootSupport

            if support_model is None:
                from supermimic.data.motion_convert import load_g1_model

                support_model = load_g1_model()
            self.foot_support = FootSupport(support_model, wm.heads.fk)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        """Average sequential actor microbatch gradients, with one optimizer step.

        Only detached critic inputs persist between microbatches. Auxiliary losses
        retain their per-microbatch reachability normalization. One microbatch is
        the original update; additional microbatches are an explicit extension.
        """
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)
        batches, metrics_by_batch = [], []
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            for _ in range(cfg.gradient_accumulation_steps):
                metrics, batch = self._actor_microbatch(buffer)
                metrics_by_batch.append(metrics)
                if batch is not None:
                    batches.append(batch)
            allreduce_grads(self.policy)
            gn = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True
            )
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
            self.opt.step()
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        if cfg.use_critic:
            for metrics, batch in zip(metrics_by_batch, batches, strict=True):
                metrics.update(self._fit_critic(batch))
        metrics = {
            key: sum(row[key] for row in metrics_by_batch) / len(metrics_by_batch)
            for key in metrics_by_batch[0]
        }
        metrics["policy_grad_norm"] = float(gn)
        metrics["actor_microbatches"] = float(cfg.gradient_accumulation_steps)
        return metrics

    def _actor_microbatch(
        self, buffer: ReplayBuffer
    ) -> tuple[dict[str, float], CriticBatch | None]:
        cfg = self.cfg
        start = (
            buffer.sample_starts(cfg.batch_size, cfg.history, cfg.reset_start_fraction)
            if cfg.reset_start_fraction
            else buffer.sample_starts(cfg.batch_size, cfg.history)
        )
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        ro = imagine(
            self.wm,
            self.policy,
            self.lib,
            start,
            cfg.horizon,
            cfg.future_frames,
            obs_noise=cfg.obs_noise,
            step_fn=self.step_fn,
            anchor_obs=cfg.anchor_obs,
            min_survival=cfg.min_survival,
        )
        feat, goals, e_root = self._critic_inputs(ro)
        if cfg.use_critic:
            final_alive = ro.survival[-1].detach() > cfg.min_survival
            final_feat = torch.where(final_alive[:, None], feat[-1], torch.zeros_like(feat[-1]))
            final_error = torch.where(
                final_alive[:, None], e_root[-1], torch.zeros_like(e_root[-1])
            )
            final_goal = torch.where(final_alive[:, None], goals[-1], torch.zeros_like(goals[-1]))
            v_H = self.critic(final_feat, final_goal, xi_norm, final_error)
            v_H = torch.where(final_alive, v_H, torch.zeros_like(v_H))
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        auxiliary_terms = None
        support_terms = None
        auxiliary_weights = (
            cfg.joint_tracking_weight,
            cfg.height_tracking_weight,
            cfg.orientation_tracking_weight,
            cfg.vertical_velocity_tracking_weight,
        )
        if (
            any(auxiliary_weights)
            or cfg.angular_velocity_tracking_weight
            or cfg.root_xy_tracking_weight
            or self.foot_support
        ):
            targets = self.lib.ref71(ro.frames[:-1].reshape(-1)).reshape_as(ro.states[1:])
            auxiliary_loss, auxiliary_terms = tracking_auxiliary(
                ro.states[1:],
                targets,
                ro.survival[:-1],
                auxiliary_weights,
                cfg.min_survival,
                dt=self.wm.spec.dt,
                orientation_mode=cfg.orientation_loss,
                joint_weights=cfg.joint_tracking_weights,
                angular_velocity_weight=cfg.angular_velocity_tracking_weight,
                root_xy_weight=cfg.root_xy_tracking_weight,
            )
            loss = loss + auxiliary_loss
            if self.foot_support is not None:
                support_loss, support_terms = self.foot_support.loss(
                    ro.states[1:],
                    targets,
                    ro.survival[:-1],
                    cfg.foot_support_height_weight,
                    cfg.foot_support_velocity_weight,
                    min_survival=cfg.min_survival,
                    dt=self.wm.spec.dt,
                    effector_position_weight=cfg.effector_position_weight,
                    effector_position_weights=(
                        cfg.effector_position_weights
                        if tuple(cfg.effector_position_weights) != (1.0,) * 4
                        else None
                    ),
                )
                loss = loss + support_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite imagined actor objective")
        (loss / cfg.gradient_accumulation_steps).backward()

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v
        if auxiliary_terms is not None:
            for name, value in zip(
                ("joint_mse", "height_mse", "orientation_error", "vertical_velocity_mse")
                + (("angular_velocity_mse",) if cfg.angular_velocity_tracking_weight else ())
                + (("root_xy_mse",) if cfg.root_xy_tracking_weight else ()),
                auxiliary_terms.detach().tolist(),
                strict=True,
            ):
                metrics[f"tracking_aux/{name}"] = value
        if support_terms is not None:
            metrics["foot_support/height_mse"] = float(support_terms[0].detach())
            metrics["foot_support/normal_velocity_mse"] = float(support_terms[1].detach())
            if len(support_terms) > 2:
                metrics["tracking_aux/effector_position_mse"] = float(support_terms[2].detach())

        batch = (
            CriticBatch(
                feat.detach(),
                goals.detach(),
                e_root.detach(),
                xi_norm.detach(),
                ro.rewards.detach(),
                ro.dhat.detach(),
                ro.survival.detach(),
            )
            if cfg.use_critic
            else None
        )
        return metrics, batch

    def _fit_critic(self, batch: CriticBatch) -> dict[str, float]:
        cfg = self.cfg
        metrics = {}
        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, e_root_d = batch.features, batch.goals, batch.root_error
                xi_norm = batch.context
                H1, n, _ = feat_d.shape
                alive = batch.survival > cfg.min_survival
                feat_d = torch.where(alive[..., None], feat_d, torch.zeros_like(feat_d))
                goals_d = torch.where(alive[..., None], goals_d, torch.zeros_like(goals_d))
                e_root_d = torch.where(alive[..., None], e_root_d, torch.zeros_like(e_root_d))
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(
                    feat_d.reshape(H1 * n, -1),
                    goals_d.reshape(H1 * n, -1),
                    xi_rep.reshape(H1 * n, -1),
                    e_root_d.reshape(H1 * n, -1),
                ).reshape(H1, n)
                values = torch.where(alive, values, torch.zeros_like(values))
                targets = gae_targets(
                    batch.rewards, batch.termination[1:], values, cfg.gamma, cfg.gae_lambda
                )  # (H, n)
                valid = alive[:-1].reshape(-1)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])[valid]
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])[valid]
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])[valid]
                X_er = e_root_d[:-1].reshape(-1, 4)[valid]
                Y = targets.reshape(-1)[valid]
                W = batch.survival[:-1].reshape(-1)[valid]
            N = Y.shape[0]
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for idx in perm.chunk(min(N, cfg.critic_minibatches)):
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                    closs = survival_weighted_mse(pred, Y[idx], W[idx])
                    if not torch.isfinite(closs):
                        raise FloatingPointError("Non-finite reachable-state critic loss")
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), cfg.critic_grad_clip, error_if_nonfinite=True
                    )
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics
