"""Rotation of training snapshots must not delete selected deployable models."""

import importlib.util
import json
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_checkpoints.py"
    spec = importlib.util.spec_from_file_location("checkpoint_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_completed_survives_rotation_and_a_higher_scoring_incomplete_model(tmp_path):
    evaluator = load_script()
    snapshot = tmp_path / "temporary-checkpoint.pt"
    snapshot.write_bytes(b"completed motion checkpoint")
    entry = {"seed": 42, "score": 0.3}
    evaluator.retain(snapshot, tmp_path, entry, "score")
    completed = evaluator.retain(snapshot, tmp_path, entry, "completed")
    snapshot.unlink()
    snapshot.write_bytes(b"higher score but incomplete checkpoint")
    scored = evaluator.retain(snapshot, tmp_path, {"seed": 42, "score": 0.4}, "score")
    snapshot.unlink()
    assert Path(completed["checkpoint"]).read_bytes() == b"completed motion checkpoint"
    assert Path(scored["checkpoint"]).read_bytes() == b"higher score but incomplete checkpoint"


def test_rewound_resume_at_same_counter_is_evaluated_as_a_different_policy(tmp_path):
    evaluator = load_script()
    first = tmp_path / "first/training/ckpt_0001000.pt"
    resumed = tmp_path / "resumed/training/ckpt_0001000.pt"
    latest = first.with_name("ckpt_latest.pt")
    assert evaluator.candidate_key(1000, 42, first) != evaluator.candidate_key(1000, 42, resumed)
    assert evaluator.candidate_key(1000, 42, first) == evaluator.candidate_key(1000, 42, latest)


def test_evaluator_waits_for_unstarted_or_running_roots(tmp_path):
    evaluator = load_script()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "run.json").write_text(json.dumps({"status": "complete"}))
    assert not evaluator.roots_finished([first, second])
    (second / "run.json").write_text(json.dumps({"status": "running"}))
    assert not evaluator.roots_finished([first, second])
    (second / "run.json").write_text(json.dumps({"status": "stopped"}))
    assert evaluator.roots_finished([first, second])


def test_serial_archive_preserves_rotated_checkpoint_and_run_identity(tmp_path):
    evaluator = load_script()
    run = tmp_path / "run"
    (run / "training").mkdir(parents=True)
    (run / "options.json").write_text(json.dumps({"options": {"command": "train", "seed": 71}}))
    source = run / "training" / "ckpt_0000500.pt"
    source.write_bytes(b"actual published checkpoint")
    evaluator.archive_checkpoints([run])
    evaluator.archive_checkpoints([run])
    archived = run / "checkpoint-archive" / source.name
    assert source.stat().st_ino == archived.stat().st_ino
    source.unlink()
    assert archived.read_bytes() == b"actual published checkpoint"
    assert evaluator.candidates([run]) == [(500, 71, archived)]
    assert evaluator.candidate_key(500, 71, source) == evaluator.candidate_key(500, 71, archived)
