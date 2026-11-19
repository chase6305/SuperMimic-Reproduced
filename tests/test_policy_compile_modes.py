"""The optional CUDA graph mode retains the flag API and has its own allowance."""

import importlib.util
from pathlib import Path

import pytest

from supermimic.cli import parser
from supermimic.memory import validate_training_budget
from supermimic.profiles import training_config


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], None),
        (["--compile-policy"], "default"),
        (["--compile-policy", "default"], "default"),
        (["--compile-policy", "reduce-overhead"], "reduce-overhead"),
    ],
)
def test_cli_mode_and_legacy_boolean_choices(flags, expected):
    args = parser().parse_args(["train", *flags])
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_actor.py"
    spec = importlib.util.spec_from_file_location("benchmark_actor_modes", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark_args = benchmark.parser().parse_args(
        ["--motion", "input.npz", "--output", "run", *flags]
    )
    assert benchmark_args.compile_policy == args.compile_policy
    cfg = training_config("smoke", Path("data"), Path("run"), compile_policy=args.compile_policy)
    assert cfg.policy_trainer.compile == expected
    estimate = validate_training_budget(cfg, 13066, 2304)
    assert estimate["components_mib"]["policy_cudagraph_workspace"] == (
        256 if expected == "reduce-overhead" else 0
    )
    legacy = training_config("smoke", Path("data"), Path("run"), compile_policy=True)
    assert legacy.policy_trainer.compile == "default"


def test_unknown_policy_compile_mode_is_rejected():
    with pytest.raises(ValueError, match="Policy compile mode"):
        training_config("smoke", Path("data"), Path("run"), compile_policy="unknown")
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--compile-policy", "unknown"])


def test_graph_mode_budget_can_require_smaller_runtime_batches():
    options = dict(
        num_envs=128,
        compile_world_model=True,
        policy_batch_size=48,
        wm_batch_size=512,
        replay_capacity=1000,
    )
    cfg = training_config("paper-width", Path("data"), Path("run"), compile_policy=True, **options)
    validate_training_budget(cfg, 13066, 2304)
    cfg = training_config(
        "paper-width", Path("data"), Path("run"), compile_policy="reduce-overhead", **options
    )
    with pytest.raises(ValueError, match="exceeding.*process budget"):
        validate_training_budget(cfg, 13066, 2304)
