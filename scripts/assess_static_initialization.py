"""Run full static-pose initialization cases and independently replay their commands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "motion", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-rss-gib", type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.max_rss_gib) or not 0.25 <= args.max_rss_gib <= 2.0:
        parser.error("--max-rss-gib must be finite and within [0.25, 2]")
    from supermimic.artifacts import sha256
    from supermimic.cpu_resources import cpu_guard
    from supermimic.deployment.evidence import read_json_evidence
    from supermimic.deployment.initialization import parse_initialization_plan
    from supermimic.resources import GIB, ResourceLimits, host_snapshot

    # Reject malformed conditions before loading a policy or allocating physics.
    plan, plan_sha = read_json_evidence(args.plan)
    parse_initialization_plan(plan)
    limits = ResourceLimits(
        max_rss_gib=args.max_rss_gib,
        reserve_ram_gib=max(12.5, 0.2 * host_snapshot()["ram_total_bytes"] / GIB),
    )
    with cpu_guard(args.output, limits=limits, max_seconds=7200):
        from supermimic.eval.static_initialization import run_static_initialization_suite

        if sha256(args.plan) != plan_sha:
            raise ValueError("Initialization plan changed after admission")
        result = run_static_initialization_suite(args.bundle, args.motion, args.plan, args.output)
    print(
        json.dumps({key: result[key] for key in ("status", "evaluated_cases", "passes")}, indent=2)
    )


if __name__ == "__main__":
    main()
