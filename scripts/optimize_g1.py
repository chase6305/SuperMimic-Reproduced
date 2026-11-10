"""Bounded sequential G1 training/evaluation on a shared GPU.

Every training attempt uses the normal guarded CLI. Resource pressure causes a
cooldown and replay-refilled resume, never termination of unrelated processes.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from supermimic.artifacts import write_json
from supermimic.resources import ResourceLimits, admit, clean_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--until", required=True, help="UTC ISO deadline, including timezone")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--motions", type=Path, default=Path("data/npz/g1"))
    p.add_argument("--chunk-hours", type=float, default=0.5)
    p.add_argument("--profile", choices=("g1", "paper-width"), default="g1")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    p.add_argument("--compile-policy", action="store_true")
    p.add_argument("--compile-world-model", action="store_true")
    p.add_argument("--wm-batches-per-update", type=int)
    p.add_argument("--policy-batch-size", type=int)
    p.add_argument("--wm-batch-size", type=int)
    p.add_argument("--replay-capacity", type=int)
    p.add_argument(
        "--skip-evaluation", action="store_true", help="Use a separate CPU checkpoint evaluator"
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-seed1", type=Path)
    p.add_argument("--pause-file", type=Path, default=Path("runs/PAUSE_OPTIMIZATION"))
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.until).timestamp()
    if deadline <= time.time() or deadline - time.time() > 8 * 3600:
        p.error("deadline must be within the next eight hours")
    if not 0.01 <= args.chunk_hours <= 1:
        p.error("chunk-hours must be within [0.01, 1]")
    if args.num_envs < 1 or args.seeds[0] == args.seeds[1]:
        p.error("Use positive environment count and two distinct seeds")
    if args.wm_batches_per_update is not None and args.wm_batches_per_update < 1:
        p.error("wm-batches-per-update must be positive")
    if args.policy_batch_size is not None and args.policy_batch_size < 1:
        p.error("policy-batch-size must be positive")
    if args.wm_batch_size is not None and args.wm_batch_size < 1:
        p.error("wm-batch-size must be positive")
    if args.replay_capacity is not None and args.replay_capacity < 12:
        p.error("replay-capacity must exceed world-model window and policy history")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "deadline_utc": args.until,
        "attempts": [],
        "evaluations": [],
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoints": {},
    }
    checkpoints = {
        args.seeds[0]: args.resume.resolve() if args.resume else None,
        args.seeds[1]: args.resume_seed1.resolve() if args.resume_seed1 else None,
    }
    last_eval_iteration = {seed: 0 for seed in args.seeds}
    child = None
    limits = ResourceLimits()
    env = clean_environment()
    try:
        attempt = 0
        stable_since = None
        while time.time() < deadline - 300:
            if args.pause_file.exists():
                record["waiting_for_resources"] = "paused for local validation"
                write_json(args.output / "session.json", record)
                stable_since = None
                time.sleep(5)
                continue
            try:
                admit(limits)
            except RuntimeError as error:
                record["waiting_for_resources"] = str(error)
                stable_since = None
                write_json(args.output / "session.json", record)
                time.sleep(10)
                continue
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since < 20:
                time.sleep(5)
                continue
            seed = args.seeds[attempt % 2]
            hours = min(args.chunk_hours, (deadline - time.time() - 300) / 3600)
            if hours < 0.01:
                break
            output = args.output / f"seed{seed}-attempt{attempt:03d}"
            command = [
                sys.executable,
                "-I",
                "-m",
                "supermimic",
                "train",
                "--profile",
                args.profile,
                "--num-envs",
                str(args.num_envs),
                "--seed",
                str(seed),
                "--hours",
                str(hours),
                "--iterations",
                "100000",
                "--motions",
                str(args.motions.resolve()),
                "--output",
                str(output.resolve()),
            ]
            if checkpoints[seed]:
                command.extend(["--checkpoint", str(checkpoints[seed])])
            if args.compile_policy:
                command.append("--compile-policy")
            if args.compile_world_model:
                command.append("--compile-world-model")
            if args.wm_batches_per_update:
                command.extend(["--wm-batches-per-update", str(args.wm_batches_per_update)])
            if args.policy_batch_size:
                command.extend(["--policy-batch-size", str(args.policy_batch_size)])
            if args.wm_batch_size:
                command.extend(["--wm-batch-size", str(args.wm_batch_size)])
            if args.replay_capacity:
                command.extend(["--replay-capacity", str(args.replay_capacity)])
            record.pop("waiting_for_resources", None)
            record["current_output"] = str(output)
            record["attempts"].append({"seed": seed, "output": str(output), "command": command})
            write_json(args.output / "session.json", record)
            print(
                f"[session] starting seed {seed}, attempt {attempt}, budget {hours:.3f} h",
                flush=True,
            )
            with (args.output / f"launcher-{attempt:03d}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                deadline_signalled = False
                while child.poll() is None:
                    if time.time() >= deadline - 120 and not deadline_signalled:
                        child.send_signal(signal.SIGINT)
                        deadline_signalled = True
                    time.sleep(5)
                exit_code = child.wait()
            result_path = output / "result.json"
            run_path = output / "run.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            run = json.loads(run_path.read_text()) if run_path.exists() else {}
            iteration = result.get("iteration", 0)
            entry = record["attempts"][-1]
            entry.update(
                exit_code=exit_code,
                result=result,
                resource_result={
                    key: run.get(key)
                    for key in ["status", "stop_reason", "peak_rss_bytes", "peak_process_gpu_mib"]
                },
            )
            candidate = output / "training/ckpt_latest.pt"
            if candidate.is_file() and result.get("status") == "complete":
                checkpoints[seed] = candidate.resolve()
                record["checkpoints"][str(seed)] = str(checkpoints[seed])
                if (
                    not args.skip_evaluation
                    and iteration - last_eval_iteration[seed] >= 500
                    and time.time() < deadline - 60
                ):
                    bundle = output / "export"
                    evaluation = output / "cpu-deploy"
                    with (output / "postprocess.log").open("w") as log:
                        exported = subprocess.run(
                            [
                                sys.executable,
                                "-I",
                                "-m",
                                "supermimic",
                                "export",
                                "--checkpoint",
                                str(candidate),
                                "--output",
                                str(bundle),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=max(1, min(120, deadline - time.time() - 40)),
                        )
                        entry["export_exit_code"] = exported.returncode
                        if exported.returncode == 0 and time.time() < deadline - 30:
                            deployed = subprocess.run(
                                [
                                    sys.executable,
                                    "-I",
                                    "-m",
                                    "supermimic",
                                    "deploy",
                                    "--bundle",
                                    str(bundle),
                                    "--motion",
                                    str(args.motions / "walk1_subject1.npz"),
                                    "--output",
                                    str(evaluation),
                                ],
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=max(1, min(120, deadline - time.time() - 15)),
                            )
                            entry["deploy_exit_code"] = deployed.returncode
                            if deployed.returncode == 0:
                                record["evaluations"].append(
                                    {
                                        "seed": seed,
                                        "iteration": iteration,
                                        "result": json.loads(
                                            (evaluation / "result.json").read_text()
                                        ),
                                    }
                                )
                                last_eval_iteration[seed] = iteration
            elif run.get("stop_reason") is None:
                record["status"] = "failed"
                record["error"] = "Non-resource worker failure; stopped for diagnosis"
                break
            write_json(args.output / "session.json", record)
            print(
                f"[session] seed {seed}: iteration {iteration}, exit {exit_code}, "
                f"stop {run.get('stop_reason')}",
                flush=True,
            )
            attempt += 1
            if attempt >= 100:
                record["status"] = "attempt_limit"
                break
            stable_since = None
            time.sleep(30 if run.get("stop_reason") else 5)
        if record["status"] == "running":
            record["status"] = "deadline_reached"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
    except Exception as error:  # noqa: BLE001 - persist controller failure and retain worker artifacts
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if record["attempts"]:
            entry = record["attempts"][-1]
            current = Path(entry["output"])
            result_path = current / "result.json"
            checkpoint = current / "training/ckpt_latest.pt"
            if result_path.is_file() and checkpoint.is_file():
                result = json.loads(result_path.read_text())
                if result.get("status") == "complete":
                    record["checkpoints"][str(entry["seed"])] = str(checkpoint.resolve())
                    entry["result"] = result
        record["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(args.output / "session.json", record)
    return 0 if record["status"] == "deadline_reached" else 2


if __name__ == "__main__":
    raise SystemExit(main())
