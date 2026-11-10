"""Sequential CPU evaluation of published checkpoints while guarded GPU training continues.

Keep the best full-clip score and the best completed motion separately. Retain
their original training checkpoints even after training rotates its own history.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from supermimic.artifacts import sha256, write_json
from supermimic.resources import ResourceLimits, ResourceUnavailable, admit, clean_environment


def candidates(roots: list[Path]):
    selected = []
    for root in roots:
        for options_path in root.rglob("options.json"):
            run = options_path.parent
            options = json.loads(options_path.read_text())["options"]
            if options.get("command") != "train":
                continue
            for checkpoint in run.glob("training/ckpt_[0-9]*.pt"):
                selected.append((int(checkpoint.stem[5:]), options["seed"], checkpoint))
            for checkpoint in run.glob("checkpoint-archive/ckpt_[0-9]*.pt"):
                selected.append((int(checkpoint.stem[5:]), options["seed"], checkpoint))
            result = run / "result.json"
            if result.is_file():
                completed = json.loads(result.read_text())
                latest = run / "training/ckpt_latest.pt"
                if completed.get("status") == "complete" and latest.is_file():
                    selected.append((completed["iteration"], options["seed"], latest))
    return sorted(set(selected))


def candidate_key(iteration: int, seed: int, checkpoint: Path):
    # Resuming an earlier best model can revisit a counter with different weights.
    # The run directory separates those branches; latest/numbered saves within
    # one run at the same counter represent the same policy.
    return iteration, seed, str(checkpoint.parent.parent.resolve())


def roots_finished(roots: list[Path]) -> bool:
    terminal = {"complete", "failed", "stopped", "interrupted"}
    found = []
    for root in roots:
        direct = root / "run.json"
        if direct.is_file():
            found.append(json.loads(direct.read_text()).get("status") in terminal)
            continue
        runs = [
            path.parent
            for path in root.rglob("options.json")
            if json.loads(path.read_text()).get("options", {}).get("command") == "train"
        ]
        if not runs:
            return False
        for run in runs:
            path = run / "run.json"
            found.append(path.is_file() and json.loads(path.read_text()).get("status") in terminal)
    return bool(found) and all(found)


def archive_checkpoints(roots: list[Path]) -> None:
    """Keep immutable numbered inodes without loading weights or allocating Torch memory."""
    for root in roots:
        for options in root.rglob("options.json"):
            if json.loads(options.read_text()).get("options", {}).get("command") != "train":
                continue
            run = options.parent
            destination = run / "checkpoint-archive"
            destination.mkdir(exist_ok=True)
            for checkpoint in run.glob("training/ckpt_[0-9]*.pt"):
                try:
                    os.link(checkpoint, destination / checkpoint.name)
                except (FileExistsError, FileNotFoundError):
                    pass


def run_candidate(checkpoint, output, motion, deadline):
    env = clean_environment()
    env["CUDA_VISIBLE_DEVICES"] = ""
    import supermimic

    implementation = output / "implementation"
    frozen = implementation / "supermimic"
    shutil.copytree(
        Path(supermimic.__file__).parent,
        frozen,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    write_json(
        output / "implementation-sha256.json",
        {str(path.relative_to(frozen)): sha256(path) for path in sorted(frozen.rglob("*.py"))},
    )
    bootstrap = "import sys; sys.path.insert(0,sys.argv[1]); from supermimic.cli import main; raise SystemExit(main(sys.argv[2:]))"
    with (output / "worker.log").open("w") as log:
        for command in (
            ["export", "--checkpoint", str(checkpoint), "--output", str(output / "export")],
            [
                "deploy",
                "--bundle",
                str(output / "export"),
                "--motion",
                str(motion),
                "--output",
                str(output / "deploy"),
            ],
        ):
            if time.time() >= deadline - 15:
                return "deadline", {}
            try:
                executed = subprocess.run(
                    [sys.executable, "-I", "-c", bootstrap, str(implementation), *command],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, min(120, deadline - time.time() - 5)),
                )
            except subprocess.TimeoutExpired:
                return "timeout", {}
            if executed.returncode:
                sidecar = output / (command[0] + ".resources.json")
                resource = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
                return "deferred" if resource.get("stop_reason") else "failed", resource
    return "complete", json.loads((output / "deploy/result.json").read_text())


def retain(snapshot: Path, output: Path, entry: dict, kind: str) -> dict:
    destination = output / f"best-{kind}-seed{entry['seed']}.pt"
    temporary = destination.with_suffix(".tmp")
    os.link(snapshot, temporary)
    os.replace(temporary, destination)
    return {**entry, "checkpoint": str(destination.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Archive checkpoints with negligible RAM while training; export/evaluate after training ends",
    )
    parser.add_argument(
        "--exit-on-complete",
        action="store_true",
        help="Exit after all training roots have finished and current checkpoints were evaluated",
    )
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.until).timestamp()
    if not time.time() < deadline <= time.time() + 8 * 3600:
        parser.error("Evaluation deadline must be within the next eight hours")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    record = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "deadline_utc": args.until,
        "evaluations": [],
        "deferred": [],
        "best_score": {},
        "best_completed": {},
    }
    if args.resume:
        record = json.loads((args.output / "session.json").read_text())
        record.update(status="running", deadline_utc=args.until)
        record.pop("finished_utc", None)
    seen = {
        candidate_key(entry["iteration"], entry["seed"], Path(entry["source"]))
        for entry in record["evaluations"]
    }
    attempts = record.get("attempts", len(record["evaluations"]) + len(record["deferred"]))
    limits = ResourceLimits(max_rss_gib=2.0)
    try:
        while time.time() < deadline - 20 and attempts < 1000:
            if args.serial:
                archive_checkpoints(args.roots)
                if not roots_finished(args.roots):
                    record["waiting_for_training"] = (
                        "Serial mode: retaining checkpoints, no Torch/SDK worker during training"
                    )
                    write_json(args.output / "session.json", record)
                    time.sleep(2)
                    continue
                record.pop("waiting_for_training", None)
            pending = [item for item in candidates(args.roots) if candidate_key(*item) not in seen]
            if not pending:
                if args.exit_on_complete and roots_finished(args.roots):
                    record["status"] = "complete"
                    break
                time.sleep(10)
                continue
            try:
                admit(limits, args.output, cuda=False)
            except ResourceUnavailable as error:
                record["waiting_for_resources"] = str(error)
                write_json(args.output / "session.json", record)
                time.sleep(20)
                continue
            iteration, seed, checkpoint = pending[0]
            attempts += 1
            target = args.output / f"seed{seed}-iter{iteration:07d}-try{attempts:03d}"
            target.mkdir()
            record.pop("waiting_for_resources", None)
            entry = {
                "seed": seed,
                "iteration": iteration,
                "source": str(checkpoint.resolve()),
                "output": str(target.resolve()),
            }
            record["current"] = entry
            record["attempts"] = attempts
            write_json(args.output / "session.json", record)
            with tempfile.TemporaryDirectory(prefix="input-", dir=target) as temporary:
                snapshot = Path(temporary) / "checkpoint.pt"
                try:
                    expected = sha256(checkpoint)
                    shutil.copyfile(checkpoint, snapshot)
                except FileNotFoundError:
                    continue  # Trainer rotated this file between discovery and opening it.
                if sha256(snapshot) != expected:
                    raise RuntimeError("Checkpoint changed during CPU evaluation snapshot")
                snapshot.chmod(0o444)
                status, result = run_candidate(snapshot, target, args.motion.resolve(), deadline)
                entry.update(status=status, checkpoint_sha256=expected)
                if status in ("deferred", "deadline"):
                    entry["resource_result"] = result
                    record["deferred"].append(entry)
                else:
                    seen.add(candidate_key(iteration, seed, checkpoint))
                    entry["result"] = result
                    record["evaluations"].append(entry)
                if status == "complete":
                    if result["checkpoint_sha256"] != expected:
                        raise RuntimeError("Exported checkpoint fingerprint differs")
                    score = result["tracking"]["normalized_return_full_clip_zero_padded"]
                    entry["score"] = score
                    if args.keep_all_checkpoints:
                        os.link(snapshot, target / "checkpoint.pt")
                        entry["checkpoint"] = str((target / "checkpoint.pt").resolve())
                    for kind in ("score", "completed"):
                        best = record["best_" + kind]
                        if kind == "completed" and not result["completed_motion"]:
                            continue
                        if str(seed) not in best or score > best[str(seed)]["score"]:
                            best[str(seed)] = retain(snapshot, args.output, entry, kind)
                    print(
                        f"[CPU eval] seed {seed}, iteration {iteration}: {result['steps']} controls, full score {score:.4f}, complete={result['completed_motion']}",
                        flush=True,
                    )
                elif status == "failed":
                    record["status"] = "failed"
                    break
            write_json(args.output / "session.json", record)
            if status == "deferred":
                time.sleep(20)
        if record["status"] == "running":
            record["status"] = "deadline_reached" if attempts < 1000 else "attempt_limit"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(args.output / "session.json", record)
    return 0 if record["status"] in ("complete", "deadline_reached") else 2


if __name__ == "__main__":
    raise SystemExit(main())
