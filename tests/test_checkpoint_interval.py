"""Save cadence is independent of optimizer/LR settings and keeps old defaults."""

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from supermimic.cli import parser
from supermimic.profiles import training_config


def test_omitted_interval_keeps_500(tmp_path):
    cfg = training_config("smoke", tmp_path, tmp_path / "out")
    assert cfg.schedule.checkpoint_interval == 500


def test_explicit_interval_changes_only_save_cadence(tmp_path):
    old = asdict(training_config("smoke", tmp_path, tmp_path / "out"))
    new = asdict(training_config("smoke", tmp_path, tmp_path / "out", checkpoint_interval=2000))
    assert new["schedule"].pop("checkpoint_interval") == 2000
    assert old["schedule"].pop("checkpoint_interval") == 500
    assert old == new


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_cadence_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        training_config("smoke", tmp_path, tmp_path / "out", checkpoint_interval=value)


def test_cli_optional_positive_integer():
    assert parser().parse_args(["train"]).checkpoint_interval is None
    assert (
        parser().parse_args(["train", "--checkpoint-interval", "2000"]).checkpoint_interval == 2000
    )
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--checkpoint-interval", "0"])


def test_actual_worker_expression_routes_cadence(tmp_path):
    import supermimic.worker as worker

    path = Path(worker.__file__)
    nodes = [
        n.value
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "cfg" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "training_config"
    ]
    assert len(nodes) == 1
    options = vars(
        parser().parse_args(["train", "--profile", "smoke", "--checkpoint-interval", "2000"])
    )
    cfg = eval(
        compile(ast.Expression(nodes[0]), str(path), "eval"),
        {
            "training_config": training_config,
            "options": options,
            "output": tmp_path / "out",
            "private_motions": tmp_path / "motion",
            "checkpoint": None,
        },
    )
    assert cfg.schedule.checkpoint_interval == 2000


def test_resume_preserves_legacy_cadence_default_unless_explicit(tmp_path):
    import torch

    old = asdict(training_config("smoke", tmp_path, tmp_path / "out"))
    old["schedule"]["checkpoint_interval"] = 123
    path = tmp_path / "saved.pt"
    torch.save({"config": old, "iteration": 1}, path)
    # Legacy profiles regenerated cadence as500 rather than inheriting this field.
    assert (
        training_config(
            "smoke", tmp_path, tmp_path / "out", resume=path
        ).schedule.checkpoint_interval
        == 500
    )
    assert (
        training_config(
            "smoke", tmp_path, tmp_path / "out", resume=path, checkpoint_interval=2000
        ).schedule.checkpoint_interval
        == 2000
    )
