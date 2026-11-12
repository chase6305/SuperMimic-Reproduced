"""Run exactly one worker under GPU/RAM/disk/time limits, recording real exit status."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import psutil

from supermimic.artifacts import write_json
from supermimic.resources import (
    GIB,
    ResourceLimits,
    ResourceUnavailable,
    admit,
    clean_environment,
    gpu_snapshot,
    host_snapshot,
    process_gpu_mib,
)
from supermimic.runtime import freeze_implementation


def supervise(
    command: list[str],
    output: Path,
    limits: ResourceLimits,
    *,
    max_seconds: float,
    cuda: bool = True,
    freeze_source: bool = False,
) -> int:
    limits.validate()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock_path = Path("runs/.gpu.lock").resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record: dict = dict(
        status="preparing",
        command=command,
        limits=asdict(limits),
        max_seconds=max_seconds,
        cuda=cuda,
        started_at=datetime.now(UTC).isoformat(),
    )
    child = None
    stop_reason = None
    stop_started = None
    peak_rss = peak_gpu = 0
    try:
        with lock_path.open("a") as lock, (output / "worker.log").open("w") as log:
            if cuda:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record["admission"] = admit(limits, output, cuda=cuda)
            if freeze_source:
                record["implementation_sha256"] = freeze_implementation(output / "implementation")
            write_json(output / "run.json", record)
            env = clean_environment()
            if cuda:
                env["CUDA_VISIBLE_DEVICES"] = str(limits.gpu_index)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            process = psutil.Process(child.pid)
            record.update(status="running", pid=child.pid)
            write_json(output / "run.json", record)
            with (output / "resources.jsonl").open("w", buffering=1) as metrics:
                while child.poll() is None:
                    try:
                        family = [process, *process.children(recursive=True)]
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        family = []
                    rss = 0
                    for member in family:
                        try:
                            rss += member.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass  # An exited compiler child must not hide the main worker's usage.
                    host = host_snapshot(output)
                    gpu = (
                        next((g for g in gpu_snapshot() if g["index"] == limits.gpu_index), None)
                        if cuda
                        else None
                    )
                    gpu_mib = process_gpu_mib({p.pid for p in family}, gpu["uuid"]) if gpu else None
                    elapsed = time.monotonic() - started
                    peak_rss = max(peak_rss, rss)
                    peak_gpu = max(peak_gpu, gpu_mib or 0)
                    sample = dict(
                        elapsed_s=elapsed,
                        rss_bytes=rss,
                        process_gpu_mib=gpu_mib,
                        gpu=gpu,
                        host=host,
                    )
                    import json

                    metrics.write(json.dumps(sample, allow_nan=False) + "\n")
                    reason = None
                    if elapsed >= max_seconds:
                        reason = "time_limit"
                    elif rss > limits.max_rss_gib * GIB:
                        reason = "process_ram_limit"
                    elif host["ram_available_bytes"] < limits.reserve_ram_gib * GIB:
                        reason = "system_ram_reserve"
                    elif host["disk_free_bytes"] < limits.reserve_disk_gib * GIB:
                        reason = "disk_reserve"
                    elif cuda and (gpu is None or gpu_mib is None):
                        reason = "gpu_monitor_unavailable"
                    elif gpu_mib is not None and gpu_mib > limits.max_gpu_mib:
                        reason = "process_gpu_limit"
                    elif gpu and gpu["free_mib"] < limits.reserve_gpu_mib:
                        reason = "system_gpu_reserve"
                    if reason and stop_reason is None:
                        stop_reason, stop_started = reason, time.monotonic()
                        print(f"[guard] stopping worker: {reason}", flush=True)
                        (output / "STOP").write_text(reason + "\n")
                        os.killpg(child.pid, signal.SIGINT)
                    if stop_started and time.monotonic() - stop_started > limits.stop_grace_seconds:
                        os.killpg(child.pid, signal.SIGKILL)
                    time.sleep(limits.poll_seconds)
            code = child.wait()
            record.update(
                status="complete"
                if code == 0 and not stop_reason
                else "stopped"
                if stop_reason
                else "failed",
                exit_code=code,
                stop_reason=stop_reason,
                elapsed_s=time.monotonic() - started,
                peak_rss_bytes=peak_rss,
                peak_process_gpu_mib=peak_gpu,
            )
            return 0 if record["status"] == "complete" else 2 if stop_reason else code or 1
    except (ResourceUnavailable, BlockingIOError) as error:
        record.update(status="stopped", stop_reason="resource_admission", error=str(error))
        print(f"[guard] resource admission deferred: {error}", flush=True)
        return 2
    except KeyboardInterrupt:
        record.update(status="interrupted", stop_reason="user_interrupt")
        return 130
    except Exception as error:  # noqa: BLE001 - parent must record admission/worker failures
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(f"[guard] {error}", flush=True)
        return 2
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=limits.stop_grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child is not None:
            record["exit_code"] = child.returncode
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        write_json(output / "run.json", record)
