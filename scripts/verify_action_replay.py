"""Verify recorded commands against saved native states, contacts and actuator forces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "motion", "run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    from supermimic.artifacts import write_json
    from supermimic.cpu_resources import cpu_guard
    from supermimic.resources import GIB, ResourceLimits, host_snapshot

    limits = ResourceLimits(
        max_rss_gib=2, reserve_ram_gib=max(12.5, 0.2 * host_snapshot()["ram_total_bytes"] / GIB)
    )
    with cpu_guard(args.output, limits=limits, max_seconds=1800):
        from supermimic.deployment.action_replay import verify_action_replay

        result = verify_action_replay(args.bundle, args.motion, args.run, args.output)
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
