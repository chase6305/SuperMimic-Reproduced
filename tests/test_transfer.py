"""New-skill initialization cannot reuse another dataset's sampler or Adam history."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig
from supermimic.profiles import training_config
from supermimic.training.loop import Trainer


def fresh_trainer():
    wm, policy, critic = [torch.nn.Linear(3, 2) for _ in range(3)]
    return SimpleNamespace(
        device="cpu",
        wm=wm,
        policy=policy,
        critic=critic,
        wm_trainer=SimpleNamespace(opt=torch.optim.Adam(wm.parameters())),
        policy_trainer=SimpleNamespace(
            opt=torch.optim.Adam(policy.parameters()),
            critic_opt=torch.optim.Adam(critic.parameters()),
        ),
        command=SimpleNamespace(bin_return=torch.ones(5), bin_memory=torch.full((5,), 0.2)),
        motion_files={"new-dance.npz": "new-hash"},
        control_contract={"control_dt": 0.02},
        iteration=0,
        num_updates=0,
        _utd_acc=0.0,
        anchor_adaptation=None,
        transfer_provenance=None,
    )


def test_transfer_preserves_weights_and_new_task_initial_state(tmp_path):
    torch.set_num_threads(1)
    source = fresh_trainer()
    source.wm(torch.randn(4, 3)).square().mean().backward()
    source.wm_trainer.opt.step()
    path = tmp_path / "source.pt"
    cfg = TrainConfig()
    torch.save(
        {
            "iteration": 42519,
            "num_updates": 16000,
            "wm": source.wm.state_dict(),
            "policy": source.policy.state_dict(),
            "critic": source.critic.state_dict(),
            "wm_opt": source.wm_trainer.opt.state_dict(),
            "motion_files": {"walk.npz": "old-hash"},
            "control_contract": source.control_contract,
            "bin_return": torch.zeros(99),
            "bin_memory": torch.zeros(99),
            "anchor_adaptation": {"source": "baseline"},
            "config": asdict(cfg),
        },
        path,
    )
    target = fresh_trainer()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="fingerprints differ"):
        Trainer.load(target, path)
    Trainer.load(target, path, transfer=True)
    for name in ("wm", "policy", "critic"):
        for a, b in zip(
            getattr(source, name).parameters(), getattr(target, name).parameters(), strict=True
        ):
            torch.testing.assert_close(a, b)
    assert not target.wm_trainer.opt.state and not target.policy_trainer.opt.state
    assert not target.policy_trainer.critic_opt.state
    assert target.iteration == target.num_updates == target._utd_acc == 0
    torch.testing.assert_close(target.command.bin_return, torch.ones(5))
    torch.testing.assert_close(target.command.bin_memory, torch.full((5,), 0.2))
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert target.transfer_provenance["source_iteration"] == 42519
    assert target.transfer_provenance["target_motion_files"] == target.motion_files
    assert target.anchor_adaptation == {"source": "baseline"}
    config = training_config(
        "smoke",
        tmp_path,
        tmp_path / "output",
        resume=path,
        iterations=100,
        transfer=True,
        training_mode="deterministic",
        tracking_objective="dynamic",
        action_clip=16,
        learning_rate=0.0001,
        wm_one_step_batches=2,
        completion_aware_sampling=True,
    )
    assert config.transfer and config.schedule.total_iterations == 100
    assert not config.env.domain_randomization and not config.env.obs_noise
    assert config.env.action_noise_std == 0 and not config.policy_trainer.obs_noise
    assert config.policy_trainer.joint_tracking_weight == 5
    assert config.policy_trainer.vertical_velocity_tracking_weight == 0.5
    assert config.policy.action_clip == 16
    assert config.wm_trainer.one_step_batches_per_update == 2
    assert config.env.completion_aware_sampling
    full = training_config(
        "smoke",
        tmp_path,
        tmp_path / "full",
        resume=path,
        iterations=100,
        transfer=True,
        tracking_objective="full-motion",
    )
    assert full.policy_trainer.orientation_loss == "geodesic"
    assert full.policy_trainer.angular_velocity_tracking_weight == 0.1
    assert full.policy_trainer.foot_support_height_weight == 0
    contact = training_config(
        "smoke",
        tmp_path,
        tmp_path / "contact",
        resume=path,
        iterations=100,
        transfer=True,
        tracking_objective="contact",
    )
    assert contact.policy_trainer.foot_support_height_weight == 500
    assert contact.policy_trainer.foot_support_velocity_weight == 0.5
    contact = training_config(
        "smoke",
        tmp_path,
        tmp_path / "contact-tuned",
        resume=path,
        iterations=100,
        transfer=True,
        tracking_objective="contact",
        foot_support_velocity_weight=5,
        effector_position_weight=10,
        joint_tracking_weight=30,
        policy_gradient_accumulation=4,
    )
    contact_metadata = torch.load(path, weights_only=False)
    contact_metadata["config"] = asdict(contact)
    contact_metadata["iteration"] = 20
    contact_path = tmp_path / "contact.pt"
    torch.save(contact_metadata, contact_path)
    inherited_contact = training_config(
        "smoke", tmp_path, tmp_path / "contact-resume", resume=contact_path, iterations=100
    )
    assert inherited_contact.policy_trainer.foot_support_height_weight == 500
    assert inherited_contact.policy_trainer.foot_support_velocity_weight == 5
    assert inherited_contact.policy_trainer.effector_position_weight == 10
    assert inherited_contact.policy_trainer.joint_tracking_weight == 30
    assert inherited_contact.policy_trainer.gradient_accumulation_steps == 4
    restored_baseline = training_config(
        "smoke",
        tmp_path,
        tmp_path / "contact-baseline",
        resume=contact_path,
        iterations=100,
        tracking_objective="baseline",
    )
    assert restored_baseline.policy_trainer.foot_support_height_weight == 0
    assert restored_baseline.policy_trainer.foot_support_velocity_weight == 0
    assert restored_baseline.policy_trainer.effector_position_weight == 0
    assert restored_baseline.policy_trainer.joint_tracking_weight == 0
    assert full.policy_trainer.reset_start_fraction == 0.5
    assert len(full.policy_trainer.joint_tracking_weights) == 29
    assert (
        config.wm_trainer.lr
        == config.policy_trainer.lr
        == config.policy_trainer.critic_lr
        == 0.0001
    )
    metadata = torch.load(path, weights_only=False)
    metadata["config"] = asdict(config)
    metadata["iteration"] = 20
    migrated = tmp_path / "task.pt"
    torch.save(metadata, migrated)
    resumed = training_config(
        "smoke", tmp_path, tmp_path / "resumed", resume=migrated, iterations=100
    )
    assert not resumed.transfer and not resumed.env.domain_randomization
    assert resumed.policy_trainer.tracking_objective == "dynamic"
    assert resumed.policy_trainer.joint_tracking_weight == 5 and resumed.policy.action_clip == 16
    assert resumed.wm_trainer.lr == resumed.policy_trainer.lr == 0.0001
    assert resumed.wm_trainer.one_step_batches_per_update == 2
    assert resumed.env.completion_aware_sampling
    disabled = training_config(
        "smoke",
        tmp_path,
        tmp_path / "disabled",
        resume=migrated,
        iterations=100,
        completion_aware_sampling=False,
    )
    assert not disabled.env.completion_aware_sampling
    with pytest.raises(ValueError, match="cumulative iteration"):
        training_config("smoke", tmp_path, tmp_path / "output", resume=path, iterations=100)
    with pytest.raises(ValueError, match="requires"):
        training_config("smoke", tmp_path, tmp_path / "output", transfer=True)
    incompatible = fresh_trainer()
    incompatible.control_contract = {"control_dt": 0.04}
    with pytest.raises(ValueError, match="control contract"):
        Trainer.load(incompatible, path, transfer=True)


def test_joint_pose_override_is_explicit_and_validated(tmp_path):
    from supermimic.cli import parser

    options = parser().parse_args(["train", "--joint-tracking-weight", "30"])
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "pose-override",
        tracking_objective="full-motion",
        joint_tracking_weight=options.joint_tracking_weight,
    )
    assert cfg.policy_trainer.joint_tracking_weight == 30
    assert cfg.policy_trainer.height_tracking_weight == 10
    assert cfg.policy_trainer.joint_tracking_weights[:6] == (2, 2, 2, 2, 4, 4)
    disabled = training_config(
        "smoke",
        tmp_path,
        tmp_path / "pose-disabled",
        tracking_objective="full-motion",
        joint_tracking_weight=0,
    )
    assert disabled.policy_trainer.joint_tracking_weight == 0
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Joint tracking weight"):
            training_config("smoke", tmp_path, tmp_path / "invalid", joint_tracking_weight=value)
        with pytest.raises(SystemExit):
            parser().parse_args(["train", f"--joint-tracking-weight={value}"])


def test_model_window_resume_override_and_replay_admission(tmp_path):
    from supermimic.cli import parser

    options = parser().parse_args(["train", "--wm-window", "16", "--wm-batch-size", "256"])
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "long-window",
        wm_window=options.wm_window,
        wm_batch_size=options.wm_batch_size,
        replay_capacity=64,
    )
    assert cfg.wm_trainer.window == 16 and cfg.wm_trainer.batch_size == 256
    source = tmp_path / "long-window.pt"
    torch.save({"config": asdict(cfg), "iteration": 20}, source)
    inherited = training_config(
        "smoke", tmp_path, tmp_path / "resume", resume=source, iterations=100
    )
    assert inherited.wm_trainer.window == 16
    overridden = training_config(
        "smoke",
        tmp_path,
        tmp_path / "short",
        resume=source,
        iterations=100,
        wm_window=8,
        replay_capacity=16,
    )
    assert overridden.wm_trainer.window == 8
    with pytest.raises(ValueError, match="Replay capacity"):
        training_config(
            "smoke",
            tmp_path,
            tmp_path / "too-short",
            resume=source,
            iterations=100,
            replay_capacity=16,
        )
    for value in (0, -1, 65, 1.5):
        with pytest.raises(ValueError, match="World-model window"):
            training_config("smoke", tmp_path, tmp_path / "invalid-window", wm_window=value)


def test_sampling_priority_resume_and_new_motion_transfer(tmp_path):
    from supermimic.cli import parser

    options = parser().parse_args(["train", "--priority-window", "10", "12"])
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "priority",
        priority_windows_s=options.priority_window,
    )
    assert cfg.env.priority_windows_s == ((10, 12),) and cfg.env.priority_reset_ratio == 0.25
    source = tmp_path / "priority.pt"
    torch.save({"config": asdict(cfg), "iteration": 20}, source)
    inherited = training_config(
        "smoke", tmp_path, tmp_path / "resume-priority", resume=source, iterations=100
    )
    assert inherited.env.priority_windows_s == ((10, 12),)
    assert inherited.env.priority_reset_ratio == 0.25
    disabled = training_config(
        "smoke",
        tmp_path,
        tmp_path / "no-priority",
        resume=source,
        iterations=100,
        priority_reset_ratio=0,
    )
    assert disabled.env.priority_reset_ratio == 0
    transferred = training_config(
        "smoke",
        tmp_path,
        tmp_path / "new-motion",
        resume=source,
        transfer=True,
        iterations=100,
    )
    assert transferred.env.priority_reset_ratio == 0 and not transferred.env.priority_windows_s
    with pytest.raises(ValueError, match="requires"):
        training_config("smoke", tmp_path, tmp_path / "missing-windows", priority_reset_ratio=0.25)


def test_policy_action_limit_is_inside_exportable_actor():
    from supermimic.policy.nets import Policy, PolicyCfg

    config = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,), action_clip=4.0)
    policy = Policy(config)
    with torch.no_grad():
        policy.decoder.mlp.net[-1].weight.zero_()
        policy.decoder.mlp.net[-1].bias.fill_(100.0)
    obs, goal = torch.randn(2, 930), torch.randn(2, 680)
    assert torch.equal(policy(obs, goal), torch.full((2, 29), 4.0))
    exported = torch.jit.trace(policy, (obs, goal))
    torch.testing.assert_close(exported(obs, goal), policy(obs, goal))


def test_transfer_provenance_tracks_the_loaded_publication(tmp_path, monkeypatch):
    from supermimic.artifacts import sha256

    source = fresh_trainer()
    path = tmp_path / "latest.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save(
        {
            "wm": source.wm.state_dict(),
            "policy": source.policy.state_dict(),
            "critic": source.critic.state_dict(),
            "iteration": 42,
            "motion_files": source.motion_files,
            "control_contract": source.control_contract,
        },
        path,
    )
    expected = sha256(path)
    torch.save({"iteration": 43}, replacement)
    original_load = torch.load

    def rotate_before_loading(source, **kwargs):
        replacement.replace(path)
        return original_load(source, **kwargs)

    monkeypatch.setattr(torch, "load", rotate_before_loading)
    target = fresh_trainer()
    Trainer.load(target, path, transfer=True)
    assert target.transfer_provenance["source_checkpoint_sha256"] == expected != sha256(path)
    assert target.transfer_provenance["source_iteration"] == 42
    for actual, original in zip(
        target.policy.parameters(), source.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, original)
