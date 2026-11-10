"""Bound CPU export/deployment/render before loading the heavy SDK modules."""

from __future__ import annotations

import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import psutil

from supermimic.artifacts import write_json
from supermimic.resources import GIB, ResourceLimits, ResourceUnavailable, admit, host_snapshot


def _process_high_water_rss() -> int | None:
    """Linux process-lifetime RSS peak; unlike family polling, excludes children."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


@contextmanager
def cpu_guard(
    output: Path, *, limits: ResourceLimits | None = None, max_seconds: float | None = None
):
    output = output.expanduser().resolve()
    limits = limits or ResourceLimits(max_rss_gib=4.5)
    existing = output.resolve().parent
    while not existing.exists():
        existing = existing.parent
    record: dict = {
        "status": "preparing",
        "limits": asdict(limits),
        "cuda_used": False,
        "peak_rss_bytes": 0,
        "stop_reason": None,
        "process_high_water_rss_start_bytes": _process_high_water_rss(),
        "process_high_water_rss_bytes": None,
        "high_water_scope": "process lifetime, excludes children; family peak sampled separately",
    }

    def high_water_exceeded() -> bool:
        current = _process_high_water_rss()
        previous = record["process_high_water_rss_bytes"]
        if previous is not None:
            current = previous if current is None else max(previous, current)
        record["process_high_water_rss_bytes"] = current
        initial = record["process_high_water_rss_start_bytes"]
        # A pre-existing lifetime peak does not establish an in-guard violation.
        exceeded = (
            current is not None
            and initial is not None
            and current > initial
            and current > limits.max_rss_gib * GIB
        )
        record["new_process_high_water_limit_exceeded"] = exceeded
        return exceeded

    stopped = threading.Event()
    watcher = None
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    started = time.monotonic()
    try:
        record["admission"] = admit(limits, existing, cuda=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        process = psutil.Process()

        def watch():
            while not stopped.is_set():
                reason = None
                try:
                    host = host_snapshot(existing)
                    family = [process, *process.children(recursive=True)]
                    rss = 0
                    for member in family:
                        try:
                            rss += member.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)
                    if rss > limits.max_rss_gib * GIB:
                        reason = "process_ram_limit"
                    elif high_water_exceeded():
                        reason = "process_ram_high_water_limit"
                    elif host["ram_available_bytes"] < limits.reserve_ram_gib * GIB:
                        reason = "system_ram_reserve"
                    elif host["disk_free_bytes"] < limits.reserve_disk_gib * GIB:
                        reason = "disk_reserve"
                    elif max_seconds is not None and time.monotonic() - started > max_seconds:
                        reason = "time_limit"
                except (OSError, psutil.Error) as error:
                    reason = f"resource_monitor_unavailable: {error}"
                if reason and not stopped.is_set():
                    record["stop_reason"] = reason
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                stopped.wait(limits.poll_seconds)

        watcher = threading.Thread(target=watch, daemon=True, name="supermimic-cpu-resources")
        watcher.start()
        record["status"] = "running"
        yield record
        record["status"] = "complete"
    except ResourceUnavailable as error:
        record.update(status="deferred", stop_reason="resource_admission", error=str(error))
        raise
    except KeyboardInterrupt:
        record["status"] = "stopped" if record["stop_reason"] else "interrupted"
        if record["stop_reason"]:
            raise RuntimeError(f"CPU operation stopped: {record['stop_reason']}") from None
        raise
    except BaseException as error:  # noqa: BLE001 - include CLI SystemExit in resource evidence
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stopped.set()
        cleanup_interrupted = False
        try:
            if watcher is not None:
                watcher.join(timeout=5)
        except KeyboardInterrupt:
            # The monitor can detect a limit just as the operation returns.
            # Preserve that stop and its evidence even if SIGINT reaches join().
            cleanup_interrupted = True
            record["status"] = "stopped" if record["stop_reason"] else "interrupted"
        finally:
            if high_water_exceeded() and record["status"] == "complete":
                record["stop_reason"] = record["stop_reason"] or "process_ram_high_water_limit"
            if record["stop_reason"] and record["status"] == "complete":
                record["status"] = "stopped"
                cleanup_interrupted = True
            if old_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            record["elapsed_s"] = time.monotonic() - started
            write_json(output.with_name(output.name + ".resources.json"), record)
        if cleanup_interrupted:
            if record["stop_reason"]:
                raise RuntimeError(f"CPU operation stopped: {record['stop_reason']}") from None
            raise KeyboardInterrupt
