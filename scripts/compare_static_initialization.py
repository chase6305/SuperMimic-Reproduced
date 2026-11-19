"""Compare completed static suite reports and list newly passed and regressed cases."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from supermimic.artifacts import write_json
    from supermimic.eval.initialization_comparison import compare_initialization_suites

    try:
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(f"Comparison output already exists: {args.output}")
        result = compare_initialization_suites(args.reports)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".supermimic-comparison-", dir=args.output.parent
            ) as directory:
                staged = Path(directory) / "comparison.json"
                write_json(staged, result)
                # Exclusive publication also preserves a racing writer's output.
                os.link(staged, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for row in result["reports"]:
        print(f"{row['checkpoint_sha256'][:12]}: {row['passes']}/{row['cases']} reported passes")
        print("  Newly passed: " + (", ".join(row["new_passes_vs_baseline"]) or "none"))
        print("  Regressions: " + (", ".join(row["regressions_vs_baseline"]) or "none"))
    print("Report comparison only; independent physics validation is separate.")


if __name__ == "__main__":
    main()
