"""Verify a complete static-pose CPU MuJoCo run using the fixed standing gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _score_inputs(inputs):
    from supermimic.data.schema import open_motion
    from supermimic.deployment.validation import validate_contract
    from supermimic.eval.static_hold import (
        judge_static_hold,
        static_hold_metrics,
        validate_static_reference,
    )
    from supermimic.interfaces import ROOT_BODY

    validate_contract(inputs.contract)
    with open_motion(inputs.motion_path) as (reference, _):
        validate_static_reference(reference)
        metrics = static_hold_metrics(
            inputs.trajectory.states,
            reference,
            inputs.body_names.index(ROOT_BODY),
            inputs.trajectory.arrays["physical_contacts"],
        )
    seconds = len(inputs.trajectory.arrays["actions"]) * inputs.trajectory.control_dt
    return metrics, judge_static_hold(metrics, inputs.completed_motion, seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "motion", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-rss-gib", type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.max_rss_gib) or not 0.25 <= args.max_rss_gib <= 2:
        parser.error("--max-rss-gib must be finite and within [0.25, 2]")
    from supermimic.cpu_resources import cpu_guard
    from supermimic.resources import GIB, ResourceLimits, host_snapshot

    limits = ResourceLimits(
        max_rss_gib=args.max_rss_gib,
        reserve_ram_gib=max(12.0, 0.2 * host_snapshot()["ram_total_bytes"] / GIB),
    )
    with cpu_guard(args.output, limits=limits):
        import supermimic.eval.static_hold as acceptance
        from supermimic.artifacts import sha256, write_json
        from supermimic.deployment.assessment import open_assessment_inputs
        from supermimic.runtime import source_manifest

        with open_assessment_inputs(args.bundle, args.motion, args.run, args.output) as inputs:
            metrics, judgement = _score_inputs(inputs)
            write_json(
                args.output,
                {
                    "checkpoint_sha256": inputs.result["checkpoint_sha256"],
                    "trajectory_sha256": inputs.trajectory.sha256,
                    "motion_sha256": inputs.result["motion_sha256"],
                    **inputs.evidence,
                    "assessment_implementation_sha256": sha256(Path(__file__)),
                    "acceptance_implementation_sha256": sha256(Path(acceptance.__file__)),
                    "assessment_implementation_package_sha256": source_manifest(),
                    "judgement": judgement,
                    "metrics": metrics,
                },
            )
        print(
            json.dumps(
                {
                    "passed": judgement["passed"],
                    "failed_checks": [
                        check["name"] for check in judgement["checks"] if not check["passed"]
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
