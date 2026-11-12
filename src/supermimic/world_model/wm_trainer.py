"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

import torch

from supermimic.config import WMTrainerCfg as WMTrainerCfg
from supermimic.data.motion_library import MotionLibrary
from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.reference import ReferenceFrame
from supermimic.world_model.reference import reference_at as reference_at
from supermimic.world_model.world_model import WorldModel


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [
            ReferenceFrame(
                None,
                None,
                None,
                None,
                r.root_pos_w[j * n : (j + 1) * n],
                r.root_quat_w[j * n : (j + 1) * n],
            )
            for j in range(k)
        ]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        specs = [(self.cfg.window, "")] * self.cfg.batches_per_update + [
            (1, "one_step/")
        ] * self.cfg.one_step_batches_per_update
        for window, prefix in specs:
            w = buffer.sample_windows(self.cfg.batch_size, window)
            loss, m = self.loss_on(w)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite world-model loss")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(
                self.wm.parameters(), self.cfg.grad_clip, error_if_nonfinite=True
            )
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                key = prefix + k
                agg[key] = agg.get(key, 0.0) + v
                counts[key] = counts.get(key, 0) + 1
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / counts[k] for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
