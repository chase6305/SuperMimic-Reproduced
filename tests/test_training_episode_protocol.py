"""Long reference episodes and their reset protocol survive strict continuation."""

from dataclasses import asdict

import pytest
import torch

from supermimic.cli import parser
from supermimic.config import TrainConfig
from supermimic.profiles import training_config


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_seed_rejects_values_the_environment_cannot_use_before_reading_checkpoint(tmp_path, seed):
    with pytest.raises(ValueError, match="Seed must be an integer"):
        training_config(
            "smoke", tmp_path, tmp_path / "invalid-seed", seed=seed, resume=tmp_path / "missing.pt"
        )


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_accepts_environment_range_boundaries(tmp_path, seed):
    assert training_config("smoke", tmp_path, tmp_path / "valid-seed", seed=seed).seed == seed
    assert parser().parse_args(["train", "--seed", str(seed)]).seed == seed


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_cli_rejects_seed_before_resource_probe_or_output_allocation(tmp_path, seed, monkeypatch):
    from supermimic import cli

    monkeypatch.setattr(
        cli, "host_snapshot", lambda: pytest.fail("probed resources for invalid seed")
    )
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        cli.main(["train", "--seed", str(seed), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_reset_only_noise_scale_overrides_mode_and_survives_resume(tmp_path):
    args = parser().parse_args(
        ["train", "--training-mode", "deterministic", "--reset-noise-scale", ".4"]
    )
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "reset-only",
        training_mode=args.training_mode,
        reset_noise_scale=args.reset_noise_scale,
    )
    assert cfg.env.reset_noise and cfg.env.reset_noise_scale == 0.4
    assert not cfg.env.obs_noise and not cfg.env.domain_randomization
    assert not cfg.policy_trainer.obs_noise and cfg.env.action_noise_std == 0
    checkpoint = tmp_path / "reset.pt"
    torch.save({"config": asdict(cfg), "iteration": 10}, checkpoint)
    restored = training_config("smoke", tmp_path, tmp_path / "restored", resume=checkpoint)
    assert restored.env.reset_noise and restored.env.reset_noise_scale == 0.4
    disabled = training_config(
        "smoke",
        tmp_path,
        tmp_path / "disabled",
        resume=checkpoint,
        training_mode="randomized",
        reset_noise_scale=0,
    )
    assert not disabled.env.reset_noise and disabled.env.reset_noise_scale == 0
    assert disabled.env.obs_noise and disabled.env.domain_randomization
    assert disabled.env.action_noise_std == 0.1
    legacy = asdict(cfg)
    legacy["env"].pop("reset_noise_scale")
    torch.save({"config": legacy, "iteration": 10}, checkpoint)
    assert (
        training_config(
            "smoke", tmp_path, tmp_path / "legacy", resume=checkpoint
        ).env.reset_noise_scale
        == 1
    )


@pytest.mark.parametrize("scale", [-1, 1.01, float("nan"), float("inf"), True])
def test_reset_noise_scale_rejects_invalid_values(tmp_path, scale):
    with pytest.raises(ValueError, match="Reset noise scale"):
        training_config("smoke", tmp_path, tmp_path / "invalid", reset_noise_scale=scale)


@pytest.mark.parametrize("scale", ["-0.1", "1.1"])
def test_cli_rejects_reset_noise_scale_outside_unit_range(scale):
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--reset-noise-scale", scale])


def test_episode_protocol_and_pose_weight_inherit_then_accept_explicit_overrides(tmp_path):
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "new",
        episode_length_s=30,
        sampling_mode="start",
        tracking_objective="contact",
        height_tracking_weight=40,
        orientation_tracking_weight=12,
        foot_support_height_weight=200,
    )
    assert cfg.env.episode_length_s == 30 and cfg.env.sampling_mode == "start"
    assert cfg.policy_trainer.orientation_tracking_weight == 12
    assert cfg.policy_trainer.height_tracking_weight == 40
    assert cfg.policy_trainer.foot_support_height_weight == 200
    checkpoint = tmp_path / "source.pt"
    torch.save({"config": asdict(cfg), "iteration": 10}, checkpoint)
    resumed = training_config("smoke", tmp_path, tmp_path / "resumed", resume=checkpoint)
    assert resumed.env.episode_length_s == 30 and resumed.env.sampling_mode == "start"
    assert resumed.policy_trainer.orientation_tracking_weight == 12
    assert resumed.policy_trainer.height_tracking_weight == 40
    assert resumed.policy_trainer.foot_support_height_weight == 200
    changed = training_config(
        "smoke",
        tmp_path,
        tmp_path / "changed",
        resume=checkpoint,
        episode_length_s=45,
        sampling_mode="uniform",
        orientation_tracking_weight=0,
        height_tracking_weight=0,
        foot_support_height_weight=0,
    )
    assert changed.env.episode_length_s == 45 and changed.env.sampling_mode == "uniform"
    assert changed.policy_trainer.orientation_tracking_weight == 0
    assert changed.policy_trainer.height_tracking_weight == 0
    assert changed.policy_trainer.foot_support_height_weight == 0
    transferred = training_config(
        "smoke", tmp_path, tmp_path / "transferred", resume=checkpoint, transfer=True
    )
    assert transferred.env.episode_length_s == TrainConfig().env.episode_length_s
    assert transferred.env.sampling_mode == "return"


def test_cli_keeps_training_protocol_out_of_evaluation(tmp_path):
    args = parser().parse_args(
        [
            "train",
            "--episode-length-seconds",
            "30",
            "--sampling-mode",
            "start",
            "--root-orientation-tracking-weight",
            "12",
            "--root-height-tracking-weight",
            "40",
            "--foot-support-height-weight",
            "200",
        ]
    )
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "run",
        episode_length_s=args.episode_length_s,
        sampling_mode=args.sampling_mode,
        orientation_tracking_weight=args.orientation_tracking_weight,
        height_tracking_weight=args.height_tracking_weight,
        foot_support_height_weight=args.foot_support_height_weight,
    )
    assert cfg.env.episode_length_s == 30 and cfg.env.sampling_mode == "start"
    assert cfg.policy_trainer.orientation_tracking_weight == 12
    assert cfg.policy_trainer.height_tracking_weight == 40
    assert cfg.policy_trainer.foot_support_height_weight == 200
    with pytest.raises(SystemExit):
        parser().parse_args(["eval", "--episode-length-seconds", "30"])


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_episode_duration_rejects_invalid_values_before_environment_allocation(tmp_path, value):
    with pytest.raises(ValueError, match="Episode length"):
        training_config("smoke", tmp_path, tmp_path / "invalid", episode_length_s=value)


def test_sampling_priorities_cannot_silently_override_start_sampling(tmp_path):
    with pytest.raises(ValueError, match="Sampling mode"):
        training_config("smoke", tmp_path, tmp_path / "bad", sampling_mode="unknown")
    with pytest.raises(ValueError, match="priorities require return"):
        training_config(
            "smoke",
            tmp_path,
            tmp_path / "bad-priority",
            sampling_mode="start",
            priority_windows_s=[(1, 2)],
            priority_reset_ratio=0.25,
        )
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "priority",
        sampling_mode="return",
        priority_windows_s=[(1, 2)],
        priority_reset_ratio=0.25,
    )
    assert cfg.env.priority_reset_ratio == 0.25


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_root_orientation_override_is_validated(tmp_path, weight):
    with pytest.raises(ValueError, match="Root orientation tracking weight"):
        training_config("smoke", tmp_path, tmp_path / "invalid", orientation_tracking_weight=weight)


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_root_height_override_is_validated(tmp_path, weight):
    with pytest.raises(ValueError, match="Root height tracking weight"):
        training_config("smoke", tmp_path, tmp_path / "invalid", height_tracking_weight=weight)


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_foot_support_height_override_is_validated(tmp_path, weight):
    with pytest.raises(ValueError, match="Foot support height weight"):
        training_config("smoke", tmp_path, tmp_path / "invalid", foot_support_height_weight=weight)


@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("anchor", [False, True])
def test_restored_anchor_feedback_is_in_the_planned_training_config(tmp_path, transfer, anchor):
    """The pre-simulator config must describe the actual inherited goal interface."""
    import json

    from supermimic.artifacts import write_json

    source = training_config("smoke", tmp_path, tmp_path / "source")
    source.env.future_frames = 7
    source.env.anchor_obs = anchor
    source.policy.goal_dim = 485 if anchor else 476
    source.policy_trainer.future_frames = 7
    source.policy_trainer.anchor_obs = anchor
    checkpoint = tmp_path / "interface.pt"
    torch.save({"config": asdict(source), "iteration": 10}, checkpoint)
    resumed = training_config(
        "smoke", tmp_path, tmp_path / "resumed", resume=checkpoint, transfer=transfer
    )
    planned_path = tmp_path / "planned-config.json"
    write_json(planned_path, asdict(resumed))
    planned = json.loads(planned_path.read_text())
    assert planned["policy"]["goal_dim"] == (485 if anchor else 476)
    assert planned["policy_trainer"]["future_frames"] == 7
    assert planned["policy_trainer"]["anchor_obs"] is anchor
