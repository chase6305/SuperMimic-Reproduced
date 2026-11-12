"""Conservative configuration admission before Warp/replay allocation.

Estimates complement measured process limits; they are not a replacement for the
supervisor, since compiler and driver allocations vary across SDKs and devices.
"""

from __future__ import annotations

from supermimic.interfaces import ACT_DIM, goal_dim, observation_dim

MIB = 1024**2


def mlp_parameters(dimensions) -> int:
    if any(size < 1 for size in dimensions):
        raise ValueError("Network dimensions must be positive")
    return sum(
        (before + 1) * after for before, after in zip(dimensions[:-1], dimensions[1:], strict=True)
    )


def estimate_training_memory(cfg, reference_frames: int) -> dict:
    if reference_frames < 1 or cfg.env.num_envs < 1:
        raise ValueError("Memory planning requires positive frame/environment counts")
    goal = goal_dim(cfg.env.future_frames, cfg.env.anchor_obs)
    parameters = sum(
        [
            mlp_parameters([133, *cfg.world_model.hidden, 70]),
            mlp_parameters([4, *cfg.world_model.term_hidden, 1]),
            mlp_parameters([goal, *cfg.policy.encoder_hidden, cfg.policy.latent_dim]),
            mlp_parameters(
                [
                    cfg.policy.latent_dim + observation_dim(cfg.policy.history),
                    *cfg.policy.decoder_hidden,
                    ACT_DIM,
                ]
            ),
            mlp_parameters([71 + goal + 33 + 4, *cfg.critic.hidden, 1]),
        ]
    )
    components = {
        "context_and_simulator_base": 768.0,
        "parameters_gradients_adam": parameters * 16 / MIB,
        "simulation_worlds": float(cfg.env.num_envs),
        "replay": cfg.env.num_envs * cfg.env.replay_capacity * 683 / MIB,
        "reference_library": reference_frames * (292 if cfg.env.compact_library else 1292) / MIB,
        "compilation_workspace": 256.0
        * (bool(cfg.policy_trainer.compile) + bool(cfg.wm_trainer.compile)),
    }
    actor = (
        cfg.policy_trainer.horizon
        * cfg.policy_trainer.batch_size
        * (
            sum(cfg.world_model.hidden)
            + sum(cfg.policy.encoder_hidden)
            + sum(cfg.policy.decoder_hidden)
        )
        * 4
        * 6
        / MIB
    )
    fitting = (
        cfg.wm_trainer.window
        * cfg.wm_trainer.batch_size
        * sum(cfg.world_model.hidden)
        * 4
        * 4
        / MIB
    )
    components["activation_workspace"] = max(actor, fitting)
    components["priority_sampling"] = (
        ((reference_frames + 49) // 50) * 16 / MIB if cfg.env.priority_reset_ratio else 0.0
    )
    # Only detached critic inputs survive each backward pass; never retain K graphs.
    components["accumulated_critic_inputs"] = (
        max(0, cfg.policy_trainer.gradient_accumulation_steps - 1)
        * (cfg.policy_trainer.horizon + 1)
        * cfg.policy_trainer.batch_size
        * (71 + goal + 33 + 4 + 3)
        * 4
        / MIB
        if cfg.policy_trainer.use_critic
        else 0.0
    )
    components["foot_support_workspace"] = (
        max(64.0, cfg.policy_trainer.horizon * cfg.policy_trainer.batch_size * 30 * 4 * 512 / MIB)
        if (
            cfg.policy_trainer.foot_support_height_weight
            or cfg.policy_trainer.foot_support_velocity_weight
            or cfg.policy_trainer.effector_position_weight
        )
        else 0.0
    )
    components["root_xy_tracking_workspace"] = (
        max(1.0, cfg.policy_trainer.horizon * cfg.policy_trainer.batch_size * 2 * 4 * 8 / MIB)
        if cfg.policy_trainer.root_xy_tracking_weight
        else 0.0
    )
    return {
        "estimated_cuda_mib": sum(components.values()),
        "parameters": parameters,
        "components_mib": components,
        "heuristic": True,
    }


def validate_training_budget(cfg, reference_frames: int, max_gpu_mib: int) -> dict:
    estimate = estimate_training_memory(cfg, reference_frames)
    if estimate["estimated_cuda_mib"] > max_gpu_mib:
        raise ValueError(
            f"Configuration estimates {estimate['estimated_cuda_mib']:.0f} MiB CUDA memory, "
            f"exceeding the {max_gpu_mib} MiB process budget; reduce environments/batches"
        )
    return estimate
