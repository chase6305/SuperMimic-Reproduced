"""GAE targets, policy objective, and a short end-to-end training smoke run."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = Path(os.environ.get("SUPERMIMIC_TEST_MOTIONS", ROOT / "data/npz/lafan1"))


def test_gae_targets_match_hand_computation():
    from supermimic.imagination.objective import gae_targets

    H, n = 4, 3
    g, lam = 0.9, 0.8
    torch.manual_seed(0)
    r = torch.rand(H, n)
    d = torch.rand(H, n) * 0.3
    V = torch.rand(H + 1, n)
    out = gae_targets(r, d, V, g, lam)
    R = V[H].clone()
    for t in range(H - 1, -1, -1):
        R = r[t] + g * (1 - d[t]) * ((1 - lam) * V[t + 1] + lam * R)
        np.testing.assert_allclose(out[t].numpy(), R.numpy(), atol=1e-6)
    # lambda = 1, d = 0 -> Monte-Carlo discounted return with bootstrap.
    out1 = gae_targets(r, torch.zeros_like(d), V, g, 1.0)
    mc = r[0] + g * r[1] + g**2 * r[2] + g**3 * r[3] + g**4 * V[4]
    np.testing.assert_allclose(out1[0].numpy(), mc.numpy(), atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz"
)
def test_training_smoke(tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name="smoke", log_root=tmp_path, seed=1)
    cfg.data.npz_dir = NPZ
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.policy_trainer.compile = None
    cfg.wm_trainer.compile = None
    cfg.schedule.total_iterations = 60
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 60
    tr = Trainer(cfg)
    try:
        tr.run()
        assert tr.num_updates > 0
        assert (tmp_path / "smoke" / "ckpt_latest.pt").exists()
        # Resume path works.
        cfg2 = TrainConfig(
            **{
                **cfg.__dict__,
                "resume": tmp_path / "smoke" / "ckpt_latest.pt",
                "run_name": "smoke2",
            }
        )
        tr.env.close()
        tr2 = Trainer(cfg2)
        assert tr2.iteration == 60
        tr2.env.close()
        # Explicit new-dataset transfer resets counters, Adam and sampling bins.
        from copy import deepcopy

        cfg3 = deepcopy(cfg)
        cfg3.resume = tmp_path / "smoke" / "ckpt_latest.pt"
        cfg3.transfer = True
        cfg3.run_name = "dance-transfer"
        cfg3.data.clips = ["dance1_subject1"]
        cfg3.env.obs_noise = cfg3.env.reset_noise = cfg3.env.domain_randomization = False
        cfg3.env.action_noise_std = 0
        cfg3.env.completion_aware_sampling = True
        cfg3.env.priority_windows_s = ((1, 3),)
        cfg3.env.priority_reset_ratio = 0.25
        cfg3.policy.action_clip = 16
        cfg3.policy_trainer.joint_tracking_weight = 5
        cfg3.policy_trainer.height_tracking_weight = 10
        cfg3.policy_trainer.orientation_tracking_weight = 2
        cfg3.policy_trainer.vertical_velocity_tracking_weight = 0.5
        cfg3.policy_trainer.angular_velocity_tracking_weight = 0.1
        cfg3.policy_trainer.foot_support_height_weight = 500
        cfg3.policy_trainer.foot_support_velocity_weight = 0.5
        cfg3.policy_trainer.effector_position_weight = 10
        cfg3.policy_trainer.gradient_accumulation_steps = 2
        cfg3.wm_trainer.one_step_batches_per_update = 1
        cfg3.wm_trainer.window = 16
        cfg3.policy_trainer.orientation_loss = "geodesic"
        cfg3.policy_trainer.reset_start_fraction = 0.5
        tr3 = Trainer(cfg3)
        assert tr3.command.cfg.completion_aware_sampling
        assert tr3.command.priority_distribution.sum().item() == pytest.approx(1)
        assert tr3.iteration == tr3.num_updates == 0 and not tr3.wm_trainer.opt.state
        assert set(tr3.motion_files) == {"dance1_subject1.npz"}
        assert tr3.transfer_provenance["source_iteration"] == 60
        tr3.run()
        assert tr3.iteration == 60 and tr3.num_updates > 0
    finally:
        pass
