"""Create a separate fixed-pose reference from a validated G1 motion frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    from supermimic.cpu_resources import cpu_guard
    from supermimic.resources import GIB, ResourceLimits, host_snapshot

    limits = ResourceLimits(
        max_rss_gib=2, reserve_ram_gib=max(12.5, 0.2 * host_snapshot()["ram_total_bytes"] / GIB)
    )
    with cpu_guard(args.output, limits=limits, max_seconds=180):
        from supermimic.data.static_reference import make_static_reference

        info = make_static_reference(
            args.source, args.output, frame=args.frame, seconds=args.seconds
        )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
