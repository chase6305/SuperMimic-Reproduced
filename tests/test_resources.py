"""Admission and worker-stop behavior without allocating a CUDA context."""

import json
import signal
import sys
from types import SimpleNamespace

import psutil
import pytest

from supermimic.resources import GIB, ResourceLimits, admit, clean_environment
from supermimic.supervisor import supervise


@pytest.fixture
def available_host(monkeypatch):
    # These tests exercise child exit/signals, independently of concurrent jobs.
    def snapshot(path):
        return {"ram_available_bytes": 100 * GIB, "disk_free_bytes": 100 * GIB}

    monkeypatch.setattr("supermimic.resources.host_snapshot", snapshot)
    monkeypatch.setattr("supermimic.supervisor.host_snapshot", snapshot)
    monkeypatch.setattr("supermimic.resources.gpu_snapshot", lambda: [])


@pytest.fixture
def vanished_worker(monkeypatch):
    class Child:
        pid = 999999999
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            assert self.returncode is not None
            return self.returncode

    child = Child()
    delivered = []
    process = SimpleNamespace(
        pid=child.pid,
        children=lambda recursive: [],
        memory_info=lambda: SimpleNamespace(rss=0),
    )

    def group_disappeared(pid, signum):
        assert pid == child.pid
        delivered.append(signum)
        if child.returncode is None:
            child.returncode = 0
        raise ProcessLookupError("worker group exited before signal delivery")

    monkeypatch.setattr("supermimic.supervisor.subprocess.Popen", lambda *a, **kw: child)
    monkeypatch.setattr("supermimic.supervisor.psutil.Process", lambda pid: process)
    monkeypatch.setattr("supermimic.supervisor.os.killpg", group_disappeared)
    return child, delivered


def test_admission_preserves_shared_gpu_headroom(monkeypatch):
    monkeypatch.setattr(
        "supermimic.resources.host_snapshot",
        lambda p: {"ram_available_bytes": 20 * GIB, "disk_free_bytes": 100 * GIB},
    )
    monkeypatch.setattr(
        "supermimic.resources.gpu_snapshot",
        lambda: [{"index": 0, "free_mib": 6000, "total_mib": 24000, "used_mib": 18000}],
    )
    with pytest.raises(RuntimeError, match="reserve"):
        admit(ResourceLimits())
    assert admit(ResourceLimits(max_gpu_mib=1500))["gpu"]["free_mib"] == 6000


def test_cpu_admission_does_not_query_gpu(available_host, monkeypatch):
    def unexpected_gpu_snapshot():
        raise AssertionError("CPU admission must not query GPU state")

    monkeypatch.setattr("supermimic.resources.gpu_snapshot", unexpected_gpu_snapshot)

    admitted = admit(ResourceLimits(), cuda=False)

    assert set(admitted) == {"host", "gpu", "limits"}
    assert admitted["gpu"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_rss_gib", -1),
        ("reserve_gpu_mib", 0),
        ("poll_seconds", float("nan")),
        ("gpu_index", -1),
    ],
)
def test_invalid_limits(field, value):
    with pytest.raises(ValueError):
        ResourceLimits(**{field: value}).validate()


def test_worker_exit_status_and_log_are_preserved(tmp_path, available_host):
    output = tmp_path / "run"
    code = supervise(
        [sys.executable, "-I", "-c", "print('worker failed'); raise SystemExit(7)"],
        output,
        ResourceLimits(poll_seconds=0.02),
        max_seconds=5,
        cuda=False,
    )
    record = json.loads((output / "run.json").read_text())
    assert code == 7 and record["exit_code"] == 7 and record["status"] == "failed"
    assert "worker failed" in (output / "worker.log").read_text()


@pytest.mark.parametrize(("exit_code", "status"), [(0, "complete"), (7, "failed")])
def test_worker_exiting_before_monitor_attach_preserves_terminal_status(
    tmp_path, available_host, vanished_worker, monkeypatch, exit_code, status
):
    child, delivered = vanished_worker
    child.returncode = exit_code

    def missing_process(pid):
        assert pid == child.pid
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr("supermimic.supervisor.psutil.Process", missing_process)
    output = tmp_path / "fast-worker"

    code = supervise(["fake-worker"], output, ResourceLimits(), max_seconds=5, cuda=False)

    record = json.loads((output / "run.json").read_text())
    assert code == record["exit_code"] == child.returncode == exit_code
    assert record["status"] == status and record["stop_reason"] is None
    assert "error" not in record
    assert not (output / "STOP").exists()
    assert (output / "resources.jsonl").read_text() == ""
    assert delivered == [signal.SIGTERM]


def test_monitor_attach_failure_is_not_ignored_for_a_running_worker(
    tmp_path, available_host, vanished_worker, monkeypatch
):
    child, delivered = vanished_worker

    def missing_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr("supermimic.supervisor.psutil.Process", missing_process)
    output = tmp_path / "monitor-unavailable"

    code = supervise(["fake-worker"], output, ResourceLimits(), max_seconds=5, cuda=False)

    record = json.loads((output / "run.json").read_text())
    assert code == 2 and record["status"] == "failed"
    assert "NoSuchProcess" in record["error"]
    assert not (output / "STOP").exists()
    assert delivered == [signal.SIGINT, signal.SIGTERM]


def test_time_limit_stops_only_own_worker(tmp_path, available_host):
    output = tmp_path / "limited"
    code = supervise(
        [sys.executable, "-I", "-c", "import time; time.sleep(5)"],
        output,
        ResourceLimits(poll_seconds=0.02, stop_grace_seconds=0.1),
        max_seconds=0.1,
        cuda=False,
    )
    record = json.loads((output / "run.json").read_text())
    assert code == 2 and record["stop_reason"] == "time_limit"
    assert record["exit_code"] != 0


def test_worker_group_exit_during_monitor_signal_preserves_time_limit(
    tmp_path, available_host, vanished_worker, monkeypatch
):
    child, delivered = vanished_worker
    ticks = iter([100.0, 200.0])
    monkeypatch.setattr("supermimic.supervisor.time.monotonic", lambda: next(ticks, 200.0))
    monkeypatch.setattr("supermimic.supervisor.time.sleep", lambda seconds: None)
    output = tmp_path / "monitor-race"

    code = supervise(["fake-worker"], output, ResourceLimits(), max_seconds=5, cuda=False)

    record = json.loads((output / "run.json").read_text())
    assert code == 2
    assert record["status"] == "stopped" and record["stop_reason"] == "time_limit"
    assert record["exit_code"] == child.returncode == 0
    assert "error" not in record
    assert (output / "STOP").read_text() == "time_limit\n"
    assert delivered == [signal.SIGINT, signal.SIGTERM]


def test_worker_group_exit_during_interrupt_cleanup_preserves_final_record(
    tmp_path, available_host, vanished_worker, monkeypatch
):
    child, delivered = vanished_worker

    def parent_interrupted(path):
        raise KeyboardInterrupt

    monkeypatch.setattr("supermimic.supervisor.host_snapshot", parent_interrupted)
    output = tmp_path / "interrupt-race"

    code = supervise(["fake-worker"], output, ResourceLimits(), max_seconds=5, cuda=False)

    record = json.loads((output / "run.json").read_text())
    assert code == 130
    assert record["status"] == "interrupted" and record["stop_reason"] == "user_interrupt"
    assert record["exit_code"] == child.returncode == 0
    assert "error" not in record
    assert delivered == [signal.SIGINT, signal.SIGTERM]


def test_python_ros_path_pollution_is_removed(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/humble/lib/python3.10/site-packages")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("MASTER_ADDR", "external-cluster-host")
    env = clean_environment()
    assert "PYTHONPATH" not in env and env["OMP_NUM_THREADS"] == "1"
    assert env["WORLD_SIZE"] == "1" and "MASTER_ADDR" not in env
