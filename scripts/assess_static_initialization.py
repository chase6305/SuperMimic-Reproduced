"""Run full static-pose initialization cases and independently replay their commands."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
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
    from supermimic.runtime import freeze_implementation, run_owned_worker

    # Reject malformed conditions before loading a policy or allocating physics.
    plan, plan_sha = read_json_evidence(args.plan)
    parse_initialization_plan(plan)
    limits = ResourceLimits(
        max_rss_gib=args.max_rss_gib,
        reserve_ram_gib=max(12.5, 0.2 * host_snapshot()["ram_total_bytes"] / GIB),
    )
    with cpu_guard(args.output, limits=limits, max_seconds=7200):
        if sha256(args.plan) != plan_sha:
            raise ValueError("Initialization plan changed after admission")
        # Run only the frozen package, even if another development turn edits the
        # checkout between cases. The suite archives that same package in output.
        bootstrap = (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from supermimic.eval.static_initialization import run_static_initialization_suite; "
            "run_static_initialization_suite(*(Path(p) for p in sys.argv[2:]))"
        )
        with tempfile.TemporaryDirectory(prefix="supermimic-static-worker-") as directory:
            implementation = Path(directory) / "implementation"
            freeze_implementation(implementation)
            command = [
                sys.executable,
                "-I",
                "-c",
                bootstrap,
                str(implementation),
                str(args.bundle),
                str(args.motion),
                str(args.plan),
                str(args.output),
            ]
            returncode = run_owned_worker(command)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        result = json.loads((args.output / "result.json").read_text())
    print(
        json.dumps({key: result[key] for key in ("status", "evaluated_cases", "passes")}, indent=2)
    )


if __name__ == "__main__":
    main()
