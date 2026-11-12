"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import math
import signal
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.artifacts import sha256, write_json
from supermimic.checkpoints import read_checkpoint, write_checkpoint
from supermimic.config import TrainConfig
from supermimic.data.discovery import discover_npz
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.interfaces import goal_dim
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import (
    allreduce_mean,
    broadcast_params,
    finalize,
    init_distributed,
)
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        # Register each owned resource as soon as it is created. Initialization
        # can fail before run() gets a chance to install its cleanup handler.
        with ExitStack() as resources:
            self._initialize(cfg, resources)
            self._resources = resources.pop_all()

    def _initialize(self, cfg: TrainConfig, resources: ExitStack) -> None:
        self.cfg = cfg
        cfg.policy.goal_dim = goal_dim(cfg.env.future_frames, cfg.env.anchor_obs)
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        cfg.policy_trainer.history = cfg.policy.history
        if cfg.env.replay_capacity <= max(cfg.wm_trainer.window + 1, cfg.policy.history + 1):
            raise ValueError("Replay capacity must exceed model window and observation history")
        if min(cfg.schedule.log_interval, cfg.schedule.checkpoint_interval) < 1:
            raise ValueError("Log and checkpoint intervals must be positive")
        self.rank, local_rank, self.world = init_distributed()
        resources.callback(finalize)
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(
                json.dumps(asdict(cfg), indent=2, default=str)
            )
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None
        if self.writer is not None:
            resources.callback(self.writer.close)

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(
                cfg.data.npz_dir,
                limit=cfg.data.limit,
                names=cfg.data.clips,
                manifest=cfg.data.manifest,
            )
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.motion_files = {path.name: sha256(path) for path in files}
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                history_length=cfg.policy.history,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                seed=cfg.seed + 1000 * self.rank,
                extra_command_kwargs={
                    "completion_aware_sampling": cfg.env.completion_aware_sampling,
                    "priority_windows_s": cfg.env.priority_windows_s,
                    "priority_reset_ratio": cfg.env.priority_reset_ratio,
                },
            ),
            device=self.device,
        )
        resources.callback(self.env.close)
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        self.control_contract = {
            "joint_names": list(robot.joint_names),
            "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
            "action_scale": action_term.scale[0].cpu().tolist(),
            "control_dt": float(env.step_dt),
        }

        fk = G1Kinematics(load_g1_model(), device=self.device)
        heads = AnalyticHeads(
            fk,
            TRACKED_BODIES,
            ANCHOR_BODY,
            ROOT_BODY,
            robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(
            self.wm,
            self.policy,
            self.critic,
            self.lib,
            cfg.policy_trainer,
            support_model=env.sim.mj_model,
        )
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(
                f"[INFO] params: world model {n_params(self.wm) / 1e6:.2f}M, policy {n_params(self.policy) / 1e6:.2f}M, critic {n_params(self.critic) / 1e6:.3f}M"
            )

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        self.stop_reason = None
        self.anchor_adaptation = None
        self.transfer_provenance = None
        if cfg.resume is not None:
            self.load(cfg.resume, transfer=cfg.transfer)
        # Replay is not persisted. Resumed absolute iterations must not bypass refill.
        self._updates_after = self.iteration + max(
            cfg.schedule.warmup_iterations, cfg.wm_trainer.window + 2, cfg.policy.history + 2
        )

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        destination = self.log_dir / name
        write_checkpoint(
            destination,
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
                "utd_acc": self._utd_acc,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "stop_reason": self.stop_reason,
                "motion_files": self.motion_files,
                "control_contract": self.control_contract,
                "anchor_adaptation": self.anchor_adaptation,
                "transfer_provenance": self.transfer_provenance,
            },
            overwrite=True,
        )
        # Keep a bounded history as well as the atomic latest file.
        for old in sorted(self.log_dir.glob("ckpt_[0-9]*.pt"))[:-3]:
            old.unlink()

    def load(self, path: Path, *, transfer: bool = False) -> None:
        # Workers snapshot checkpoints first. Map immutable CPU storage instead of
        # materializing every optimizer tensor before moving the needed state to CUDA.
        loaded = read_checkpoint(path, mmap=True, fingerprint=transfer)
        ck = loaded.data
        if (
            not transfer
            and ck.get("motion_files") is not None
            and ck["motion_files"] != self.motion_files
        ):
            raise ValueError(
                "Checkpoint motion fingerprints differ; sampler state cannot be reused"
            )
        if (
            ck.get("control_contract") is not None
            and ck["control_contract"] != self.control_contract
        ):
            raise ValueError("Checkpoint G1 control contract differs from the current environment")
        if transfer:
            if (
                self.iteration
                or self.num_updates
                or any(
                    opt.state
                    for opt in (
                        self.wm_trainer.opt,
                        self.policy_trainer.opt,
                        self.policy_trainer.critic_opt,
                    )
                )
            ):
                raise ValueError("Weight transfer requires a fresh trainer")
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.anchor_adaptation = ck.get("anchor_adaptation")
        if transfer:
            self.transfer_provenance = {
                "source_checkpoint_sha256": loaded.sha256,
                "source_iteration": ck["iteration"],
                "source_motion_files": ck.get("motion_files"),
                "target_motion_files": self.motion_files,
                "initialization": "world-model, actor and critic weights; fresh optimizers, sampler, counters and new-seed RNG",
            }
            print(f"[INFO] transferred network weights from {path}; new training starts at 0")
            return
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        self._utd_acc = ck.get("utd_acc", 0.0)
        self.transfer_provenance = ck.get("transfer_provenance")
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        old_handlers = {}

        def request_stop(signum, frame):
            self.stop_reason = signal.Signals(signum).name

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, request_stop)
            self._run()
        finally:
            try:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
            finally:
                self.close()

    def close(self) -> None:
        """Release the simulator, event writer and distributed state exactly once."""
        self._resources.close()

    def _run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            if self.stop_reason:
                break
            if sch.max_hours is not None and time.time() - t_start >= sch.max_hours * 3600:
                self.stop_reason = "training_time_limit"
                break
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Policy produced non-finite actions")
            obs, rew, term, tout, extras = env.step(action)
            if not torch.isfinite(rew).all():
                raise FloatingPointError("Simulator produced non-finite rewards")
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if (
                self.iteration >= self._updates_after
                and self.buffer.window_mask(cfg.wm_trainer.window).any()
            ):
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(
                        self.iteration,
                        sch.total_iterations,
                        cfg.wm_trainer.lr,
                        sch.lr_final_ratio,
                        sch.lr_decay_fraction,
                    )
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    invalid = {
                        f"{stage}/{key}": value
                        for stage, metrics in (("world_model", m_wm), ("policy", m_pi))
                        for key, value in metrics.items()
                        if not math.isfinite(value)
                    }
                    if invalid:
                        raise FloatingPointError(f"Non-finite training metrics: {invalid}")
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar(
                        "train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it
                    )
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if (
                        self.is_main
                        and self.buffer.size >= cfg.wm_trainer.batch_size
                        and self.buffer.window_mask(8).any()
                    ):
                        err = self.wm_trainer.online_error(
                            self.buffer, n=min(128, cfg.wm_trainer.batch_size), k=8
                        )
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar(
                        "sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it
                    )
                    self.writer.add_scalar(
                        "sampler/bin_return_mean", float(self.command.bin_return.mean()), it
                    )
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed / 3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan')) / max(agg_n, 1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan')) / max(agg_n, 1):.3f} surv {agg.get('policy/imag_survival_H', float('nan')) / max(agg_n, 1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                    progress = {
                        "iteration": it,
                        "updates": self.num_updates,
                        "elapsed_s": elapsed,
                        "episode_norm_return": tr if math.isfinite(tr) else None,
                        "metrics": {k: v / (agg_n if k != "lr" else 1) for k, v in agg.items()}
                        if agg_n
                        else {},
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "online_prediction_nse": err.cpu().tolist() if err is not None else None,
                        "time_unix_s": time.time(),
                    }
                    write_json(self.log_dir / "progress.json", progress)
                    with (self.log_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(progress, allow_nan=False) + "\n")
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(
                        stop
                    )  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
