from pathlib import Path

import pytest

from supermimic.memory import validate_training_budget
from supermimic.profiles import training_config


def test_measured_paper_width_configuration_fits_conservative_budget():
    cfg = training_config(
        "paper-width", Path("data"), Path("run"), num_envs=32, compile_policy=True
    )
    estimate = validate_training_budget(cfg, 13066, 2304)
    assert 15_000_000 < estimate["parameters"] < 20_000_000
    assert estimate["estimated_cuda_mib"] < 2304


def test_large_environment_request_rejected_before_simulator_allocation():
    cfg = training_config("g1", Path("data"), Path("run"), num_envs=4096)
    with pytest.raises(ValueError, match="exceeding.*process budget"):
        validate_training_budget(cfg, 13066, 2304)


def test_additional_compiler_has_its_own_workspace_allowance():
    cfg = training_config(
        "paper-width",
        Path("data"),
        Path("run"),
        num_envs=128,
        compile_policy=True,
        compile_world_model=True,
        wm_batches_per_update=10,
    )
    assert cfg.wm_trainer.compile == "default"
    assert cfg.wm_trainer.batches_per_update == 10
    estimate = validate_training_budget(cfg, 13066, 2304)
    assert estimate["components_mib"]["compilation_workspace"] == 512


def test_larger_policy_batch_requires_balancing_simulation_budget():
    cfg = training_config(
        "paper-width",
        Path("data"),
        Path("run"),
        num_envs=192,
        compile_policy=True,
        compile_world_model=True,
        policy_batch_size=48,
    )
    assert cfg.policy_trainer.batch_size == 48
    validate_training_budget(cfg, 13066, 2304)
    cfg.env.num_envs = 256
    with pytest.raises(ValueError, match="exceeding"):
        validate_training_budget(cfg, 13066, 2304)


def test_larger_model_fit_and_original_replay_length_fit_with_fewer_worlds():
    cfg = training_config(
        "paper-width",
        Path("data"),
        Path("run"),
        num_envs=128,
        compile_policy=True,
        compile_world_model=True,
        policy_batch_size=48,
        wm_batch_size=512,
        replay_capacity=1000,
    )
    assert cfg.env.replay_capacity == 1000
    assert cfg.wm_trainer.batch_size == 512
    validate_training_budget(cfg, 13066, 2304)
    cfg.wm_trainer.batch_size = 1024
    with pytest.raises(ValueError, match="exceeding"):
        validate_training_budget(cfg, 13066, 2304)


def test_support_geometry_requires_allowance_before_cuda_allocation():
    cfg = training_config(
        "paper-width",
        Path("data"),
        Path("run"),
        num_envs=128,
        compile_policy=True,
        compile_world_model=True,
        policy_batch_size=48,
        wm_batch_size=512,
        replay_capacity=512,
    )
    validate_training_budget(cfg, 13066, 2304)
    cfg.policy_trainer.foot_support_height_weight = 500
    with pytest.raises(ValueError, match="exceeding"):
        validate_training_budget(cfg, 13066, 2304)
    cfg.policy_trainer.batch_size = 32
    estimate = validate_training_budget(cfg, 13066, 2304)
    assert estimate["components_mib"]["foot_support_workspace"] > 0
    cfg.policy_trainer.batch_size = 48
    cfg.policy_trainer.foot_support_height_weight = 0
    cfg.policy_trainer.effector_position_weight = 10
    with pytest.raises(ValueError, match="exceeding"):
        validate_training_budget(cfg, 13066, 2304)
    cfg.policy_trainer.batch_size = 32
    validate_training_budget(cfg, 13066, 2304)
    baseline = validate_training_budget(cfg, 13066, 2304)
    cfg.policy_trainer.gradient_accumulation_steps = 4
    accumulated = validate_training_budget(cfg, 13066, 2304)
    assert accumulated["components_mib"]["accumulated_critic_inputs"] > 0
    assert (
        accumulated["components_mib"]["activation_workspace"]
        == baseline["components_mib"]["activation_workspace"]
    )
    cfg.wm_trainer.window = 16
    cfg.wm_trainer.batch_size = 256
    longer = validate_training_budget(cfg, 13066, 2304)
    assert longer["estimated_cuda_mib"] == accumulated["estimated_cuda_mib"]
