"""Explicit local budgets; paper network widths are available without paper-scale batches."""

from __future__ import annotations

import math
from pathlib import Path

from supermimic.config import CriticCfg, PolicyCfg, StateSpecCfg, TrainConfig, WorldModelCfg


def training_config(
    profile: str,
    motions: Path,
    output: Path,
    *,
    num_envs: int | None = None,
    iterations: int = 100000,
    checkpoint_interval: int | None = None,
    hours: float = 1.0,
    seed: int = 0,
    clips: list[str] | None = None,
    resume: Path | None = None,
    compile_policy: bool = False,
    compile_world_model: bool = False,
    wm_batches_per_update: int | None = None,
    policy_batch_size: int | None = None,
    policy_horizon: int | None = None,
    policy_update_scope: str | None = None,
    wm_batch_size: int | None = None,
    replay_capacity: int | None = None,
    transfer: bool = False,
    action_clip: float | None = None,
    tracking_objective: str | None = None,
    training_mode: str | None = None,
    learning_rate: float | None = None,
    wm_one_step_batches: int | None = None,
    completion_aware_sampling: bool | None = None,
    foot_support_velocity_weight: float | None = None,
    effector_position_weight: float | None = None,
    effector_position_weights: tuple[float, ...] | None = None,
    joint_tracking_weight: float | None = None,
    root_xy_tracking_weight: float | None = None,
    use_critic: bool | None = None,
    policy_gradient_accumulation: int | None = None,
    wm_window: int | None = None,
    priority_windows_s: list[tuple[float, float]] | None = None,
    priority_reset_ratio: float | None = None,
) -> TrainConfig:
    """Resolve profile defaults, checkpoint inheritance and explicit run overrides.

    Architecture and data interfaces are inherited for both resume modes. Runtime
    budgets remain selected for this run; strict continuation additionally restores
    its horizon, update scope and reset priorities. Explicit overrides apply last.
    """
    if transfer and resume is None:
        raise ValueError("Transfer requires a source checkpoint")
    cfg = TrainConfig(
        run_name="training", log_root=output, seed=seed, resume=resume, transfer=transfer
    )
    cfg.data.npz_dir = motions
    cfg.data.clips = clips
    cfg.env.num_envs = num_envs or (8 if profile == "smoke" else 32)
    cfg.env.replay_capacity = 128
    cfg.schedule.total_iterations = iterations
    cfg.schedule.max_hours = hours
    cfg.schedule.warmup_iterations = 24
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 500
    if checkpoint_interval is not None:
        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise ValueError("Checkpoint interval must be a positive integer")
        cfg.schedule.checkpoint_interval = checkpoint_interval
    _apply_profile(cfg, profile)
    if cfg.env.num_envs < 1 or iterations < 1 or hours <= 0:
        raise ValueError("Environment count, iterations and hours must be positive")
    if compile_policy:
        cfg.policy_trainer.compile = "default"
    if compile_world_model:
        cfg.wm_trainer.compile = "default"
    if wm_batches_per_update is not None:
        if wm_batches_per_update < 1:
            raise ValueError("World-model batches per update must be positive")
        cfg.wm_trainer.batches_per_update = wm_batches_per_update
    if policy_batch_size is not None:
        if policy_batch_size < 1:
            raise ValueError("Policy batch size must be positive")
        cfg.policy_trainer.batch_size = policy_batch_size
    if wm_batch_size is not None:
        if wm_batch_size < 1:
            raise ValueError("World-model batch size must be positive")
        cfg.wm_trainer.batch_size = wm_batch_size
    if replay_capacity is not None:
        if replay_capacity <= max(cfg.wm_trainer.window + 1, cfg.policy.history + 1):
            raise ValueError("Replay capacity must exceed world-model window and policy history")
        cfg.env.replay_capacity = replay_capacity
    if resume is not None:
        # Keep checkpoint I/O behind the resume boundary; new profiles stay SDK-free.
        from supermimic.checkpoints import read_checkpoint

        checkpoint = read_checkpoint(resume, mmap=True).data
        if not transfer and checkpoint["iteration"] >= iterations:
            raise ValueError("--iterations must exceed the checkpoint's cumulative iteration")
        _restore_checkpoint_config(cfg, checkpoint["config"], transfer=transfer)
    if priority_windows_s is not None:
        cfg.env.priority_windows_s = priority_windows_s
        if priority_reset_ratio is None:
            cfg.env.priority_reset_ratio = 0.25
    if priority_reset_ratio is not None:
        cfg.env.priority_reset_ratio = priority_reset_ratio
    from supermimic.envs.mdp.sampling import validate_priorities

    cfg.env.priority_windows_s = validate_priorities(
        cfg.env.priority_windows_s, cfg.env.priority_reset_ratio
    )
    if completion_aware_sampling is not None:
        cfg.env.completion_aware_sampling = completion_aware_sampling
    if action_clip is not None:
        if not math.isfinite(action_clip) or action_clip <= 0:
            raise ValueError("Action clipping must be finite and positive")
        cfg.policy.action_clip = action_clip
    if tracking_objective is not None:
        _apply_tracking_objective(cfg, tracking_objective)
    if effector_position_weights is not None:
        cfg.policy_trainer.effector_position_weights = tuple(effector_position_weights)
    if len(cfg.policy_trainer.effector_position_weights) != 4 or any(
        not math.isfinite(v) or v <= 0 for v in cfg.policy_trainer.effector_position_weights
    ):
        raise ValueError("Effector position weights must contain four finite positive values")
    cfg.policy_trainer.effector_position_weights = tuple(
        float(v) for v in cfg.policy_trainer.effector_position_weights
    )
    if policy_update_scope is not None:
        cfg.policy_trainer.update_scope = policy_update_scope
    if cfg.policy_trainer.update_scope not in ("all", "decoder", "action-head"):
        raise ValueError("Policy update scope must be all, decoder or action-head")
    if policy_horizon is not None:
        cfg.policy_trainer.horizon = policy_horizon
    if type(cfg.policy_trainer.horizon) is not int or not 1 <= cfg.policy_trainer.horizon <= 64:
        raise ValueError("Policy horizon must be an integer within [1, 64]")
    if wm_window is not None:
        if not isinstance(wm_window, int) or not 1 <= wm_window <= 64:
            raise ValueError("World-model window must be an integer within [1, 64]")
        cfg.wm_trainer.window = wm_window
    if policy_gradient_accumulation is not None:
        if (
            not isinstance(policy_gradient_accumulation, int)
            or not 1 <= policy_gradient_accumulation <= 16
        ):
            raise ValueError("Policy gradient accumulation must be an integer within [1, 16]")
        cfg.policy_trainer.gradient_accumulation_steps = policy_gradient_accumulation
    if joint_tracking_weight is not None:
        if not math.isfinite(joint_tracking_weight) or joint_tracking_weight < 0:
            raise ValueError("Joint tracking weight must be finite and nonnegative")
        cfg.policy_trainer.joint_tracking_weight = joint_tracking_weight
    if root_xy_tracking_weight is not None:
        if not math.isfinite(root_xy_tracking_weight) or root_xy_tracking_weight < 0:
            raise ValueError("Root XY tracking weight must be finite and nonnegative")
        cfg.policy_trainer.root_xy_tracking_weight = root_xy_tracking_weight
    if effector_position_weight is not None:
        if not math.isfinite(effector_position_weight) or effector_position_weight < 0:
            raise ValueError("Effector position weight must be finite and nonnegative")
        cfg.policy_trainer.effector_position_weight = effector_position_weight
    if foot_support_velocity_weight is not None:
        if not math.isfinite(foot_support_velocity_weight) or foot_support_velocity_weight < 0:
            raise ValueError("Foot support velocity weight must be finite and nonnegative")
        cfg.policy_trainer.foot_support_velocity_weight = foot_support_velocity_weight
    if use_critic is not None:
        if not isinstance(use_critic, bool):
            raise ValueError("Critic switch must be boolean")
        cfg.policy_trainer.use_critic = use_critic
    if not isinstance(cfg.policy_trainer.use_critic, bool):
        raise ValueError("Saved critic switch must be boolean")
    if training_mode is not None:
        if training_mode not in ("randomized", "deterministic"):
            raise ValueError("Unknown training mode")
        randomized = training_mode == "randomized"
        cfg.env.obs_noise = cfg.env.reset_noise = cfg.env.domain_randomization = randomized
        cfg.env.action_noise_std = 0.1 if randomized else 0.0
        cfg.policy_trainer.obs_noise = randomized
    if learning_rate is not None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        cfg.wm_trainer.lr = cfg.policy_trainer.lr = cfg.policy_trainer.critic_lr = learning_rate
    if wm_one_step_batches is not None:
        if wm_one_step_batches < 0:
            raise ValueError("Single-step batch count must be nonnegative")
        cfg.wm_trainer.one_step_batches_per_update = wm_one_step_batches
    _validate_replay_capacity(cfg)
    return cfg


def _apply_profile(cfg: TrainConfig, profile: str) -> None:
    """Apply network widths and local learning budgets without touching run identity."""
    cfg.wm_trainer.batch_size = 32
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 16
    cfg.policy_trainer.critic_epochs = 1
    cfg.policy_trainer.critic_minibatches = 2
    # Compilation is an explicit opt-in: compiler memory is substantial on a shared GPU.
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.compile = None
    if profile == "smoke":
        cfg.world_model.hidden = (128, 128)
        cfg.policy.encoder_hidden = (128, 64)
        cfg.policy.decoder_hidden = (128, 64)
        cfg.policy_trainer.horizon = 8
    elif profile == "g1":
        cfg.world_model.hidden = (512, 512, 512)
        cfg.policy.encoder_hidden = (512, 256, 128)
        cfg.policy.decoder_hidden = (512, 256, 128)
        cfg.policy_trainer.horizon = 32
        # The initial shared-GPU probe used < 900 MiB. Increase learning batches
        # inside the unchanged 2304 MiB process budget, with full-run monitoring.
        cfg.env.replay_capacity = 256
        cfg.wm_trainer.batch_size = 128
        cfg.wm_trainer.batches_per_update = 4
        cfg.policy_trainer.batch_size = 64
        cfg.policy_trainer.critic_epochs = 3
        cfg.policy_trainer.critic_minibatches = 4
    elif profile == "paper-width":
        cfg.env.replay_capacity = 256
        cfg.policy_trainer.horizon = 32
        cfg.wm_trainer.batch_size = 128
        cfg.wm_trainer.batches_per_update = 4
        cfg.policy_trainer.batch_size = 32
        cfg.policy_trainer.critic_epochs = 3
        cfg.policy_trainer.critic_minibatches = 4
    else:
        raise ValueError(f"Unknown training profile: {profile}")


def _restore_checkpoint_config(cfg: TrainConfig, saved: dict, *, transfer: bool) -> None:
    """Restore the legacy field selection while leaving this run's budgets intact."""
    cfg.policy = PolicyCfg(**saved["policy"])
    cfg.critic = CriticCfg(**saved["critic"])
    wm = dict(saved["world_model"])
    wm["state_spec"] = StateSpecCfg(**wm["state_spec"])
    cfg.world_model = WorldModelCfg(**wm)
    cfg.env.anchor_obs = saved["env"].get("anchor_obs", False)
    cfg.env.completion_aware_sampling = saved["env"].get("completion_aware_sampling", False)
    if not transfer:
        cfg.env.priority_windows_s = saved["env"].get("priority_windows_s", ())
        cfg.env.priority_reset_ratio = saved["env"].get("priority_reset_ratio", 0.0)
    cfg.env.future_frames = saved["env"]["future_frames"]
    cfg.policy_trainer.future_frames = cfg.env.future_frames
    cfg.policy_trainer.history = cfg.policy.history
    if not transfer:
        cfg.policy_trainer.horizon = saved["policy_trainer"].get(
            "horizon", cfg.policy_trainer.horizon
        )
    if not transfer:
        cfg.policy_trainer.update_scope = saved["policy_trainer"].get("update_scope", "all")
    for name in ("obs_noise", "reset_noise", "domain_randomization", "action_noise_std"):
        if name in saved["env"]:
            setattr(cfg.env, name, saved["env"][name])
    cfg.policy_trainer.obs_noise = saved["policy_trainer"].get("obs_noise", cfg.env.obs_noise)
    cfg.wm_trainer.lr = saved["wm_trainer"]["lr"]
    cfg.wm_trainer.window = saved["wm_trainer"].get("window", cfg.wm_trainer.window)
    cfg.wm_trainer.one_step_batches_per_update = saved["wm_trainer"].get(
        "one_step_batches_per_update", 0
    )
    cfg.policy_trainer.lr = saved["policy_trainer"]["lr"]
    cfg.policy_trainer.critic_lr = saved["policy_trainer"]["critic_lr"]
    for name in (
        "use_critic",
        "tracking_objective",
        "joint_tracking_weight",
        "root_xy_tracking_weight",
        "height_tracking_weight",
        "orientation_tracking_weight",
        "vertical_velocity_tracking_weight",
        "angular_velocity_tracking_weight",
        "foot_support_height_weight",
        "foot_support_velocity_weight",
        "effector_position_weight",
        "effector_position_weights",
        "orientation_loss",
        "joint_tracking_weights",
        "reset_start_fraction",
        "gradient_accumulation_steps",
    ):
        if name in saved["policy_trainer"]:
            setattr(cfg.policy_trainer, name, saved["policy_trainer"][name])


def _apply_tracking_objective(cfg: TrainConfig, tracking_objective: str) -> None:
    """Reset the objective preset before individual weight overrides are applied."""
    presets = {
        "baseline": (0.0, 0.0, 0.0, 0.0),
        "pose": (5.0, 10.0, 2.0, 0.0),
        "dynamic": (5.0, 10.0, 2.0, 0.5),
        "full-motion": (10.0, 10.0, 2.0, 0.5),
        "contact": (10.0, 10.0, 2.0, 0.5),
    }
    if tracking_objective not in presets:
        raise ValueError("Unknown tracking objective")
    cfg.policy_trainer.tracking_objective = tracking_objective
    (
        cfg.policy_trainer.joint_tracking_weight,
        cfg.policy_trainer.height_tracking_weight,
        cfg.policy_trainer.orientation_tracking_weight,
        cfg.policy_trainer.vertical_velocity_tracking_weight,
    ) = presets[tracking_objective]
    cfg.policy_trainer.orientation_loss = (
        "geodesic" if tracking_objective in ("full-motion", "contact") else "chordal"
    )
    cfg.policy_trainer.angular_velocity_tracking_weight = (
        0.1 if tracking_objective in ("full-motion", "contact") else 0.0
    )
    cfg.policy_trainer.joint_tracking_weights = (
        (2.0, 2.0, 2.0, 2.0, 4.0, 4.0) * 2 + (1.0,) * 17
        if tracking_objective in ("full-motion", "contact")
        else (1.0,) * 29
    )
    cfg.policy_trainer.reset_start_fraction = (
        0.5 if tracking_objective in ("full-motion", "contact") else 0.0
    )
    cfg.policy_trainer.foot_support_height_weight = (
        500.0 if tracking_objective == "contact" else 0.0
    )
    cfg.policy_trainer.foot_support_velocity_weight = (
        0.5 if tracking_objective == "contact" else 0.0
    )
    cfg.policy_trainer.effector_position_weight = 0.0
    cfg.policy_trainer.effector_position_weights = (1.0,) * 4
    cfg.policy_trainer.root_xy_tracking_weight = 0.0


def _validate_replay_capacity(cfg: TrainConfig) -> None:
    """Validate with the final inherited/overridden model window and history."""
    if cfg.env.replay_capacity <= max(cfg.wm_trainer.window + 1, cfg.policy.history + 1):
        raise ValueError("Replay capacity must exceed model window and observation history")
