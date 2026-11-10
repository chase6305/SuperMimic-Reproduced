"""Summarize actual run outcomes and observed resources without loading checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from supermimic.artifacts import write_json


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def summarize(root: Path) -> dict:
    training, deployment, checks = [], [], []
    for path in sorted(root.rglob("run.json")):
        run = read(path)
        result = read(path.with_name("result.json"))
        options = read(path.with_name("options.json")).get("options", {})
        peak_gpu = run.get("peak_process_gpu_mib", 0)
        peak_rss = run.get("peak_rss_bytes", 0)
        resources = path.with_name("resources.jsonl")
        if resources.is_file():
            with resources.open() as stream:
                for line in stream:
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n") and run["status"] == "running":
                            break
                        raise
                    peak_gpu = max(peak_gpu, sample.get("process_gpu_mib") or 0)
                    peak_rss = max(peak_rss, sample.get("rss_bytes") or 0)
        entry = {
            "path": str(path.parent),
            "status": run.get("status"),
            "exit_code": run.get("exit_code"),
            "stop_reason": run.get("stop_reason"),
            "observed_peak_gpu_mib": peak_gpu,
            "observed_peak_rss_bytes": peak_rss,
            "elapsed_s": run.get("elapsed_s"),
        }
        if options.get("command") == "train" or "iteration" in result:
            entry.update(
                profile=options.get("profile"),
                seed=options.get("seed"),
                iteration=result.get("iteration"),
                updates=result.get("updates"),
                checkpoint_sha256=result.get("checkpoint_sha256"),
            )
            for target, final, start in [
                ("new_iterations", "iteration", "starting_iteration"),
                ("new_updates", "updates", "starting_updates"),
            ]:
                entry[target] = (
                    result[final] - result[start] if final in result and start in result else None
                )
            training.append(entry)
        else:
            checks.append(entry)
    for path in sorted(root.rglob("result.json")):
        result = read(path)
        if result.get("backend") == "mujoco_cpu":
            deployment.append({"path": str(path.parent), **result})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "training": training,
        "deployment": deployment,
        "checks": checks,
        "max_observed_process_gpu_mib": max(
            (entry["observed_peak_gpu_mib"] for entry in training + checks), default=0
        ),
        "scope": "Run status and full-motion completion are separate. Cumulative training iterations are not summed across resumes.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.root)
    write_json(args.output, report)
    print(
        f"{len(report['training'])} training records, {len(report['deployment'])} deployment records; "
        f"max observed GPU {report['max_observed_process_gpu_mib']} MiB"
    )


if __name__ == "__main__":
    main()
