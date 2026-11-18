"""Run a declared suite of full static references with fixed acceptance limits."""

from __future__ import annotations

import json
from pathlib import Path

from supermimic.artifacts import sha256, snapshot_file, write_json
from supermimic.deployment.evidence import read_json_evidence
from supermimic.deployment.initialization import parse_initialization_plan


def run_static_initialization_suite(bundle: Path, motion: Path, plan: Path, output: Path) -> dict:
    """Called inside the CLI's CPU guard and the caller's shared heavy-work lock.

    Every case runs the entire supplied motion without resetting. A failed
    quality gate remains in the report; evidence or replay errors stop the suite
    with a partial report. Files are pinned before any policy is evaluated.
    """
    from supermimic.data.schema import open_motion
    from supermimic.deployment.action_replay import verify_action_replay
    from supermimic.deployment.assessment import open_assessment_inputs
    from supermimic.deployment.mujoco import deploy
    from supermimic.eval.static_hold import (
        judge_static_hold,
        static_hold_metrics,
        validate_static_reference,
    )
    from supermimic.interfaces import ROOT_BODY
    from supermimic.runtime import freeze_implementation

    bundle, motion, plan, output = map(Path, (bundle, motion, plan, output))
    declared, plan_sha = read_json_evidence(plan)
    cases = parse_initialization_plan(declared)
    # Resource estimates and schema admission precede publication and inference.
    with open_motion(motion) as (reference, info):
        validate_static_reference(reference)
        seconds = (info["frames"] - 1) / info["fps"]
        if not 30 <= seconds <= 600 or seconds * len(cases) > 3600:
            raise ValueError("Static suite needs 30..600 seconds per case and at most 3600 total")
    output.mkdir(parents=True, exist_ok=False)
    snapshot_file(plan, output / "plan.json", expected_sha256=plan_sha)
    contract, contract_sha = read_json_evidence(bundle / "contract.json")
    from supermimic.deployment.validation import validate_contract

    validate_contract(contract)
    if (
        not isinstance(contract.get("files"), dict)
        or not {"model.mjb", "policy.ts"} <= contract["files"].keys()
        or any(
            not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}
            for name in contract["files"]
        )
    ):
        raise ValueError(
            "Bundle evidence filenames must be local basenames and include model/policy"
        )
    snapshot_file(
        bundle / "contract.json", output / "bundle/contract.json", expected_sha256=contract_sha
    )
    for name, digest in contract["files"].items():
        snapshot_file(bundle / name, output / "bundle" / name, expected_sha256=digest)
    pinned_motion = output / "reference.npz"
    snapshot_file(motion, pinned_motion)
    with open_motion(pinned_motion) as (reference, info):
        validate_static_reference(reference)
        # Re-admit the pinned publication if the source pathname changed.
        seconds = (info["frames"] - 1) / info["fps"]
        if not 30 <= seconds <= 600 or seconds * len(cases) > 3600:
            raise ValueError("Pinned reference exceeds the static suite duration budget")
    implementation = freeze_implementation(output / "implementation")
    report = {
        "status": "running",
        "schema": declared["schema"],
        "plan_sha256": plan_sha,
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "contract_sha256": contract_sha,
        "motion_sha256": sha256(pinned_motion),
        "seconds_per_case": seconds,
        "declared_cases": len(cases),
        "records": [],
        "implementation_package_sha256": implementation,
        "implementation_snapshot": str(output / "implementation"),
        "scope": "Only the declared initial conditions and full supplied reference in CPU MuJoCo. No guarantee for untested perturbations, longer durations or hardware.",
    }
    write_json(output / "result.json", report)
    try:
        for name, perturbation in cases:
            run = output / "cases" / name
            deploy(output / "bundle", pinned_motion, run, initialization=perturbation)
            with open_assessment_inputs(
                output / "bundle", pinned_motion, run, run / "assessment.json"
            ) as inputs:
                with open_motion(inputs.motion_path) as (reference, _):
                    metrics = static_hold_metrics(
                        inputs.trajectory.states,
                        reference,
                        inputs.body_names.index(ROOT_BODY),
                        inputs.trajectory.arrays["physical_contacts"],
                    )
                duration = len(inputs.trajectory.arrays["actions"]) * inputs.trajectory.control_dt
                judgement = judge_static_hold(metrics, inputs.completed_motion, duration)
                assessment = {
                    "metrics": metrics,
                    "judgement": judgement,
                    **inputs.evidence,
                    "initialization": perturbation.as_dict(),
                    "initialization_sha256": inputs.result["initialization_sha256"],
                }
            write_json(run / "assessment.json", assessment)
            replay = verify_action_replay(
                output / "bundle", pinned_motion, run, run / "replay.json"
            )
            write_json(run / "replay.json", replay)
            report["records"].append(
                {
                    "id": name,
                    "initialization": perturbation.as_dict(),
                    "assessment": assessment,
                    "replay": replay,
                    "result_sha256": sha256(run / "result.json"),
                }
            )
            write_json(output / "result.json", report)
            if not replay["passed"]:
                raise ValueError(f"Independent action replay differs for {name}")
            print(
                json.dumps(
                    {
                        "case": name,
                        "passed": judgement["passed"],
                        "root_xy_rmse_m": metrics["root_xy_rmse_m"],
                    }
                ),
                flush=True,
            )
        report.update(
            status="complete",
            evaluated_cases=len(cases),
            passes=sum(r["assessment"]["judgement"]["passed"] for r in report["records"]),
            all_replays_bitexact=True,
        )
        return report
    except BaseException as error:  # noqa: BLE001 - preserve interrupted evidence
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "result.json", report)
