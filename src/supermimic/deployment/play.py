"""Interactive policy playback through the guarded mjlab process."""

from __future__ import annotations

import torch


class DictPolicy:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def play_checkpoint(checkpoint, files, num_envs, viewer):
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    policy, cfg = load_policy_from_checkpoint(checkpoint, "cuda:0")
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files],
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        history_length=cfg["policy"]["history"],
        episode_length_s=1.0e5,
        action_noise_std=0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=True,
    )
    env = RslRlVecEnvWrapper(make_env(args, device="cuda:0"))
    try:
        factory = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
        factory(env, DictPolicy(policy)).run()
    finally:
        env.close()
