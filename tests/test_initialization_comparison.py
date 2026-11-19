"""Like-for-like comparisons surface regressions even when total passes increase."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

from supermimic.deployment.initialization import InitialPerturbation
from supermimic.eval.initialization_comparison import compare_initialization_suites
from supermimic.eval.static_hold import judge_static_hold


@pytest.fixture
def reports(tmp_path):
    cases = [
        ("baseline", {}),
        ("shift", {"root_translation_m": [0.01, 0, 0]}),
        ("pitch", {"root_world_rotation_vector_rad": [0, 0.01, 0]}),
        ("joint", {"joint_position_noise_std_rad": 0.01, "seed": 1}),
    ]
    plan = json.dumps(
        {
            "schema": "g1-static-initialization-v1",
            "cases": [{"id": name, "perturbation": value} for name, value in cases],
        }
    )
    paths = []
    for model, passing in [("a", {"baseline", "pitch"}), ("b", {"baseline", "shift", "joint"})]:
        folder = tmp_path / model
        folder.mkdir()
        (folder / "plan.json").write_text(plan)
        rows = []
        for name, perturbation in cases:
            metrics = {
                "root_xy_rmse_m": 0.01 if name in passing else 0.2,
                "joint_rmse_rad": 0.01,
                "root_height_rmse_m": 0.01,
                "root_orientation_rmse_rad": 0.01,
                "nonfoot_ground_force_substeps": 0,
                "joint_velocity_rms_rad_s": 0.01,
            }
            rows.append(
                {
                    "id": name,
                    "initialization": InitialPerturbation.from_mapping(perturbation).as_dict(),
                    "assessment": {
                        "metrics": metrics,
                        "model_sha256": "c" * 64,
                        "judgement": judge_static_hold(metrics, True, 60),
                    },
                    "replay": {
                        "passed": True,
                        "fields_bitexact": {
                            "states": True,
                            "physical_contacts": True,
                            "actuator_forces": True,
                        },
                    },
                }
            )
        report = {
            "schema": "g1-static-initialization-v1",
            "status": "complete",
            "records": rows,
            "plan_sha256": hashlib.sha256(plan.encode()).hexdigest(),
            "checkpoint_sha256": model * 64,
            "motion_sha256": "d" * 64,
            "seconds_per_case": 60,
            "all_replays_bitexact": True,
            "declared_cases": 4,
            "evaluated_cases": 4,
            "passes": len(passing),
        }
        path = folder / "result.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_more_total_passes_with_a_regression_is_not_a_strict_case_improvement(reports):
    result = compare_initialization_suites(reports)
    row = result["reports"][1]
    assert row["passes"] == 3 and result["reports"][0]["passes"] == 2
    assert row["new_passes_vs_baseline"] == ["joint", "shift"]
    assert row["regressions_vs_baseline"] == ["pitch"]
    assert not row["strict_case_coverage_improvement"]
    assert result["case_matrix"][2] == {"id": "pitch", "passes": [True, False]}


@pytest.mark.parametrize("change", ["reference", "plan", "gate", "partial", "count", "replay"])
def test_inconsistent_or_incomparable_reports_are_rejected(reports, change):
    path = reports[1]
    report = json.loads(path.read_text())
    if change == "reference":
        report["motion_sha256"] = "e" * 64
    elif change == "plan":
        path.with_name("plan.json").write_text("{}")
    elif change == "gate":
        report["records"][0]["assessment"]["judgement"]["criteria"]["max_root_xy_rmse_m"] = 0.5
    elif change == "partial":
        report["status"] = "running"
    elif change == "count":
        report["passes"] = 4
    else:
        report["records"][0]["replay"]["fields_bitexact"]["states"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare_initialization_suites(reports)


def run_comparison_cli(monkeypatch, reports, output):
    script = Path(__file__).resolve().parents[1] / "scripts/compare_static_initialization.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "--reports", *map(str, reports), "--output", str(output)]
    )
    runpy.run_path(str(script), run_name="__main__")


@pytest.mark.parametrize("filename", ["result.json", "plan.json"])
def test_cli_output_cannot_overwrite_input_evidence(reports, monkeypatch, filename):
    output = reports[0].with_name(filename)
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        run_comparison_cli(monkeypatch, reports, output)
    assert error.value.code == 2
    assert output.read_bytes() == before


def test_cli_publishes_a_new_report_and_removes_staging(reports, monkeypatch, tmp_path):
    output = tmp_path / "new/comparison.json"
    run_comparison_cli(monkeypatch, reports, output)
    assert json.loads(output.read_text()) == compare_initialization_suites(reports)
    assert not list(output.parent.glob(".supermimic-comparison-*"))


def test_cli_preserves_a_racing_writers_report(reports, monkeypatch, tmp_path):
    output = tmp_path / "comparison.json"
    link = os.link

    def publish_first(source, destination):
        destination.write_bytes(b"another writer's evidence")
        link(source, destination)

    monkeypatch.setattr(os, "link", publish_first)
    with pytest.raises(SystemExit) as error:
        run_comparison_cli(monkeypatch, reports, output)
    assert error.value.code == 2
    assert output.read_bytes() == b"another writer's evidence"
    assert not list(tmp_path.glob(".supermimic-comparison-*"))
