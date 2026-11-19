"""Compare like-for-like completed static suite reports without re-running physics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from supermimic.deployment.initialization import InitialPerturbation, parse_initialization_plan
from supermimic.eval.static_hold import judge_static_hold


def _read(path: Path, budget: int):
    with path.open("rb") as stream:
        payload = stream.read(budget + 1)
    if len(payload) > budget:
        raise ValueError(f"Comparison JSON exceeds its byte budget: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Comparison needs JSON objects: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _suite(path: Path):
    report, digest = _read(path, 4 * 1024**2)
    if report.get("status") != "complete":
        raise ValueError("Comparison requires complete suite reports")
    plan, plan_digest = _read(path.with_name("plan.json"), 1024**2)
    cases = parse_initialization_plan(plan)
    if report.get("plan_sha256") != plan_digest or report.get("schema") != plan["schema"]:
        raise ValueError("Comparison report differs from its pinned condition plan")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != len(cases):
        raise ValueError("Comparison needs every declared case exactly once")
    if any(
        type(report.get(k)) is not int or report[k] != len(cases)
        for k in ("evaluated_cases", "declared_cases")
    ):
        raise ValueError("Comparison case counts disagree")
    if report.get("all_replays_bitexact") is not True:
        raise ValueError("Comparison requires completed recorded replay checks")
    seconds = report.get("seconds_per_case")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 30 <= seconds <= 600:
        raise ValueError("Comparison requires the declared full reference duration")
    for name in ("checkpoint_sha256", "motion_sha256"):
        value = report.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"Comparison requires a valid {name}")
    passes = []
    for (name, perturbation), row in zip(cases, records, strict=True):
        if row.get("id") != name:
            raise ValueError("Comparison case order or identity differs from the plan")
        # JSON lists and dataclass tuples are normalized by the same public parser.
        declared = InitialPerturbation.from_mapping(row.get("initialization"))
        if declared != perturbation:
            raise ValueError("Comparison initialization differs from the plan")
        assessment = row["assessment"]
        metrics = assessment["metrics"]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in metrics.values()
        ):
            raise ValueError("Comparison requires finite nonnegative reported metrics")
        judgement = assessment["judgement"]
        checks = {c["name"]: c for c in judgement["checks"]}
        complete = checks["completed_motion"]["value"]
        duration = checks["duration_s"]["value"]
        if (
            type(complete) is not bool
            or type(duration) not in (int, float)
            or not 0 <= duration <= seconds + 1e-6
        ):
            raise ValueError("Comparison completion or reached duration is invalid")
        expected = judge_static_hold(metrics, complete, duration)
        if type(judgement.get("passed")) is not bool or judgement != expected:
            raise ValueError("Comparison judgement differs from the unchanged fixed quality gate")
        replay = row["replay"]
        fields = replay["fields_bitexact"]
        if (
            replay.get("passed") is not True
            or set(fields) != {"states", "physical_contacts", "actuator_forces"}
            or any(v is not True for v in fields.values())
        ):
            raise ValueError("Comparison requires successful recorded native replay checks")
        if complete and not math.isclose(duration, seconds, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Comparison completed case did not reach the declared full duration")
        passes.append(judgement["passed"])
    if type(report.get("passes")) is not int or report["passes"] != sum(passes):
        raise ValueError("Comparison pass count disagrees with the fixed case judgements")
    models = {row["assessment"]["model_sha256"] for row in records}
    if len(models) != 1:
        raise ValueError("Comparison requires a consistent model within each suite")
    report["model_sha256"] = models.pop()
    return report, digest, [name for name, _ in cases], passes


def compare_initialization_suites(paths: list[Path]) -> dict:
    """Surface case regressions, rather than ranking solely by total pass count.

    This is a bounded metadata comparison. Judgements are checked against the
    fixed gate and pinned plan; trajectories are not loaded or replayed again.
    """
    if not 2 <= len(paths) <= 8:
        raise ValueError("Comparison requires two to eight completed suite reports")
    try:
        loaded = [_suite(Path(path)) for path in paths]
    except (KeyError, TypeError, AttributeError):
        raise ValueError("Comparison suite report has incomplete or malformed fields") from None
    first, _, names, baseline_passes = loaded[0]
    required = {name for name, passed in zip(names, baseline_passes, strict=True) if passed}
    rows = []
    for path, (report, digest, case_names, passes) in zip(paths, loaded, strict=True):
        if case_names != names or any(
            report[k] != first[k]
            for k in ("schema", "plan_sha256", "motion_sha256", "model_sha256", "seconds_per_case")
        ):
            raise ValueError(
                "Comparison requires identical conditions, reference and full duration"
            )
        passed_ids = {name for name, passed in zip(names, passes, strict=True) if passed}
        rows.append(
            {
                "report": str(path),
                "report_sha256": digest,
                "checkpoint_sha256": report["checkpoint_sha256"],
                "cases": len(names),
                "passes": len(passed_ids),
                "new_passes_vs_baseline": sorted(passed_ids - required),
                "regressions_vs_baseline": sorted(required - passed_ids),
                "strict_case_coverage_improvement": passed_ids > required,
            }
        )
    return {
        "scope": "Reported suite metadata and fixed-gate consistency only; no new physics validation or generalization claim.",
        "plan_sha256": first["plan_sha256"],
        "motion_sha256": first["motion_sha256"],
        "seconds_per_case": first["seconds_per_case"],
        "reports": rows,
        "case_matrix": [
            {"id": name, "passes": [data[3][i] for data in loaded]} for i, name in enumerate(names)
        ],
    }
