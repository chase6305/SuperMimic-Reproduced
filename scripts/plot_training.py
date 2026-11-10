"""Plot observed training, full-motion evaluation and resources from run artifacts.

This reads JSON/JSONL only, never model weights or CUDA. Resumed chunks stay
separate lines so refilling replay is not presented as uninterrupted training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise
                return  # Last line may be in flight in a live run.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, help="Separate CPU evaluator session.json")
    args = parser.parse_args()
    from supermimic.cpu_resources import cpu_guard

    with cpu_guard(args.output):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from supermimic.artifacts import write_json

        figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        titles = (
            "Training episode return (symlog; 5-sample average)",
            "World-model 8-step prediction NSE",
            "CPU full-motion coverage (no reset)",
            "CPU full-motion zero-padded return",
            "Observed process GPU allocation",
            "Observed process-tree RAM",
        )
        for axis, title in zip(axes.flat, titles, strict=True):
            axis.set_title(title)
            axis.grid(alpha=0.2)
        records = []
        colors = {}
        labelled = set()
        for metrics in sorted(args.root.rglob("training/metrics.jsonl")):
            run = metrics.parent.parent
            options = json.loads((run / "options.json").read_text())["options"]
            seed = options["seed"]
            if seed not in colors:
                colors[seed] = f"C{len(colors) % 10}"
            color = colors[seed]
            samples = list(rows(metrics))
            if not samples:
                continue
            label = f"seed {seed}" if seed not in labelled else None
            labelled.add(seed)
            x = [r["iteration"] for r in samples]
            smoothed = []
            for index in range(len(samples)):
                values = [
                    r["episode_norm_return"]
                    for r in samples[max(0, index - 4) : index + 1]
                    if r.get("episode_norm_return") is not None
                ]
                smoothed.append(sum(values) / len(values) if values else float("nan"))
            axes[0, 0].plot(x, smoothed, color=color, label=label, linewidth=1)
            axes[0, 1].plot(
                x,
                [
                    r["online_prediction_nse"][-1]
                    if r.get("online_prediction_nse")
                    else float("nan")
                    for r in samples
                ],
                color=color,
                linewidth=1,
            )
            resource_file = run / "resources.jsonl"
            limits = json.loads((run / "run.json").read_text())["limits"]
            axes[2, 0].axhline(limits["max_gpu_mib"], color=color, linestyle="--", alpha=0.4)
            axes[2, 1].axhline(limits["max_rss_gib"], color=color, linestyle="--", alpha=0.4)
            resources = list(rows(resource_file)) if resource_file.is_file() else []
            if resources:
                start = resources[0]["elapsed_s"]
                seconds = [(r["elapsed_s"] - start) / 60 for r in resources]
                axes[2, 0].plot(
                    seconds,
                    [r.get("process_gpu_mib") or 0 for r in resources],
                    color=color,
                    alpha=0.7,
                    linewidth=1,
                )
                axes[2, 1].plot(
                    seconds,
                    [r["rss_bytes"] / 1024**3 for r in resources],
                    color=color,
                    alpha=0.7,
                    linewidth=1,
                )
            evaluation_file = run / "cpu-deploy/result.json"
            evaluation = (
                json.loads(evaluation_file.read_text()) if evaluation_file.is_file() else {}
            )
            if evaluation.get("status") == "complete" and evaluation.get("tracking"):
                evaluation_iteration = json.loads((run / "result.json").read_text())["iteration"]
                tracking = evaluation["tracking"]
                axes[1, 0].scatter(
                    evaluation_iteration, tracking["clip_coverage"], color=color, s=25
                )
                axes[1, 1].scatter(
                    evaluation_iteration,
                    tracking["normalized_return_full_clip_zero_padded"],
                    color=color,
                    s=25,
                )
            records.append(
                {
                    "run": str(run),
                    "seed": seed,
                    "last_logged_iteration": x[-1],
                    "last_logged_episode_return": samples[-1].get("episode_norm_return"),
                    "cpu_evaluation": evaluation,
                }
            )
        if not records:
            raise ValueError("No training/metrics.jsonl found below root")
        if args.evaluations:
            evaluated = json.loads(args.evaluations.read_text())
            for entry in evaluated["evaluations"]:
                result = entry.get("result", {})
                if entry.get("status") != "complete" or not result.get("tracking"):
                    continue
                seed = entry["seed"]
                if seed not in colors:
                    colors[seed] = f"C{len(colors) % 10}"
                tracking = result["tracking"]
                axes[1, 0].scatter(
                    entry["iteration"], tracking["clip_coverage"], color=colors[seed], s=20
                )
                axes[1, 1].scatter(
                    entry["iteration"],
                    tracking["normalized_return_full_clip_zero_padded"],
                    color=colors[seed],
                    s=20,
                )
        axes[0, 0].legend()
        # Preserve large negative failure episodes without flattening ordinary
        # returns into an apparently constant zero line.
        axes[0, 0].set_yscale("symlog", linthresh=1)
        for axis in axes[:2].flat:
            axis.set_xlabel("Cumulative iteration")
        for axis in axes[2]:
            axis.set_xlabel("Minutes within each run chunk")
        axes[2, 0].set_ylabel("MiB")
        axes[2, 1].set_ylabel("GiB")
        axes[1, 0].set_ylim(0, 1.05)
        figure.suptitle("SuperMimic G1: training scores and continuous deployment are separate")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)
        write_json(
            args.output.with_suffix(".json"),
            {
                "root": str(args.root),
                "runs": records,
                "evaluations": str(args.evaluations) if args.evaluations else None,
            },
        )
    print(args.output)


if __name__ == "__main__":
    main()
