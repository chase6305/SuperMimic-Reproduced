import json
import os
import time

import pytest

from supermimic.cpu_resources import cpu_guard
from supermimic.resources import GIB, ResourceLimits, ResourceUnavailable


def test_low_memory_admission_never_enters_heavy_operation(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ResourceUnavailable("available RAM below reserve")

    monkeypatch.setattr("supermimic.cpu_resources.admit", unavailable)
    entered = False
    with pytest.raises(ResourceUnavailable):
        with cpu_guard(tmp_path / "bundle"):
            entered = True
    assert not entered
    record = json.loads((tmp_path / "bundle.resources.json").read_text())
    assert record["status"] == "deferred"


def test_cpu_guard_records_peak_and_restores_device_environment(tmp_path, monkeypatch):
    monkeypatch.setattr("supermimic.cpu_resources.admit", lambda *a, **kw: {})
    monkeypatch.setattr(
        "supermimic.cpu_resources.host_snapshot",
        lambda p: {"ram_available_bytes": 100 * GIB, "disk_free_bytes": 100 * GIB},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with cpu_guard(tmp_path / "operation", limits=ResourceLimits(poll_seconds=0.005)):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        time.sleep(0.02)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
    record = json.loads((tmp_path / "operation.resources.json").read_text())
    assert record["status"] == "complete" and record["peak_rss_bytes"] > 0


def test_cpu_time_budget_interrupts_own_operation_and_records_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("supermimic.cpu_resources.admit", lambda *a, **kw: {})
    monkeypatch.setattr(
        "supermimic.cpu_resources.host_snapshot",
        lambda p: {"ram_available_bytes": 100 * GIB, "disk_free_bytes": 100 * GIB},
    )
    with pytest.raises(RuntimeError, match="time_limit"):
        with cpu_guard(
            tmp_path / "limited", limits=ResourceLimits(poll_seconds=0.005), max_seconds=0.02
        ):
            time.sleep(0.5)
    record = json.loads((tmp_path / "limited.resources.json").read_text())
    assert record["status"] == "stopped" and record["stop_reason"] == "time_limit"


def test_cli_system_exit_is_recorded_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("supermimic.cpu_resources.admit", lambda *a, **kw: {})
    monkeypatch.setattr(
        "supermimic.cpu_resources.host_snapshot",
        lambda p: {"ram_available_bytes": 100 * GIB, "disk_free_bytes": 100 * GIB},
    )
    with pytest.raises(SystemExit):
        with cpu_guard(tmp_path / "failed-conversion"):
            raise SystemExit(2)
    record = json.loads((tmp_path / "failed-conversion.resources.json").read_text())
    assert record["status"] == "failed"


def test_late_monitor_interrupt_during_cleanup_still_records_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("supermimic.cpu_resources.admit", lambda *a, **kw: {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    class LateMonitor:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            raise KeyboardInterrupt

    monkeypatch.setattr("supermimic.cpu_resources.threading.Thread", LateMonitor)
    with pytest.raises(RuntimeError, match="process_ram_limit"):
        with cpu_guard(tmp_path / "late") as record:
            record["stop_reason"] = "process_ram_limit"
    saved = json.loads((tmp_path / "late.resources.json").read_text())
    assert saved["status"] == "stopped" and saved["stop_reason"] == "process_ram_limit"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"


@pytest.mark.parametrize(
    "initial_peak,final_peak,stopped", [(64, 160, True), (64, 64, False), (200, 200, False)]
)
def test_cleanup_detects_new_peak_between_family_polls(
    tmp_path, monkeypatch, initial_peak, final_peak, stopped
):
    monkeypatch.setattr("supermimic.cpu_resources.admit", lambda *a, **kw: {})
    peaks = iter([initial_peak * 2**20, final_peak * 2**20])
    monkeypatch.setattr("supermimic.cpu_resources._process_high_water_rss", lambda: next(peaks))

    class PassiveMonitor:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

    monkeypatch.setattr("supermimic.cpu_resources.threading.Thread", PassiveMonitor)

    def operation():
        with cpu_guard(tmp_path / "peak", limits=ResourceLimits(max_rss_gib=0.125)):
            pass

    if stopped:
        with pytest.raises(RuntimeError, match="process_ram_high_water_limit"):
            operation()
    else:
        operation()
    record = json.loads((tmp_path / "peak.resources.json").read_text())
    assert record["status"] == ("stopped" if stopped else "complete")
    assert record["new_process_high_water_limit_exceeded"] is stopped
