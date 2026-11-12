"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.checkpoints import policy_from_checkpoint, read_checkpoint
from supermimic.policy.nets import Policy


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)

    @property
    def requested_steps(self) -> int:
        return self.frames - 1

    @property
    def coverage(self) -> float:
        return self.steps / max(self.requested_steps, 1)

    @property
    def full_clip_return(self) -> float:
        """Assign zero reward to the unexecuted portion after an early failure."""
        return self.norm_return * self.coverage


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = read_checkpoint(path).data
    policy = policy_from_checkpoint(ck).to(device)
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(
    policy: Policy,
    npz_files: list[str],
    device: str = "cuda:0",
    num_envs: int | None = None,
    future_frames: int = 10,
    verbose: bool = True,
    anchor_obs: bool = False,
) -> list[ClipResult]:
    if not npz_files:
        raise ValueError("Benchmark needs at least one motion")
    if num_envs is not None and num_envs < 1:
        raise ValueError("Benchmark environment count must be positive")
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        history_length=policy.cfg.history,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            # reset() already advances the command to frame 1. Execute exactly
            # frames-1 controls, rather than waiting for the delayed clip timeout
            # to execute additional controls against the clamped final target.
            horizon = steps >= lib.clip_len[current.clamp(max=n_clips - 1)] - 1
            done = (term | tout | horizon) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={
                                n: float(term_sum[e, i]) / max(st, 1)
                                for i, n in enumerate(term_names)
                            },
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                manual = ids[~(term[ids] | tout[ids]) & (nxt[ids] < n_clips)]
                if manual.numel():
                    obs, _ = env.reset(env_ids=manual)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {
        "clips": n,
        "failed": failed,
        "mean_norm_return": mean_ret,
        "step_weighted_norm_return": w_ret,
        "mean_full_clip_zero_padded_return": sum(r.full_clip_return for r in results) / max(n, 1),
        "mean_clip_coverage": sum(r.coverage for r in results) / max(n, 1),
        "completed_clips": sum(not r.failed and r.steps == r.requested_steps for r in results),
        "return_protocol": "norm_return scores executed steps; full_clip_zero_padded_return includes failed remainder as zero",
    }
    if results and results[0].terms:
        out["terms"] = {
            k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1)
            for k in results[0].terms
        }
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "summary": summarize(results),
                "clips": [
                    {
                        **asdict(r),
                        "requested_steps": r.requested_steps,
                        "clip_coverage": r.coverage,
                        "full_clip_zero_padded_return": r.full_clip_return,
                    }
                    for r in results
                ],
            },
            indent=1,
        )
    )
