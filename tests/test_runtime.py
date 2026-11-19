"""Frozen CLI execution retains assets and ignores ambient Python package paths."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from supermimic.runtime import (
    freeze_implementation,
    isolated_cli_command,
    run_owned_worker,
    source_manifest,
)


def test_source_manifest_tracks_relative_python_paths_and_content_changes(tmp_path):
    package = tmp_path / "package"
    (package / "deployment").mkdir(parents=True)
    root_source = package / "__init__.py"
    nested_source = package / "deployment/reference.py"
    root_source.write_text('"""Package."""\n')
    nested_source.write_text("REFERENCE = 1\n")
    (package / "deployment/asset.txt").write_text("non-Python package asset")

    initial = source_manifest(package)

    assert initial == {
        "__init__.py": hashlib.sha256(root_source.read_bytes()).hexdigest(),
        "deployment/reference.py": hashlib.sha256(nested_source.read_bytes()).hexdigest(),
    }
    nested_source.write_text("REFERENCE = 2\n")
    changed = source_manifest(package)
    assert changed["__init__.py"] == initial["__init__.py"]
    assert changed["deployment/reference.py"] != initial["deployment/reference.py"]
    assert (
        changed["deployment/reference.py"] == hashlib.sha256(nested_source.read_bytes()).hexdigest()
    )


def test_freeze_retains_package_assets_and_omits_bytecode_from_copy_and_manifest(tmp_path):
    source = tmp_path / "source"
    (source / "licenses").mkdir(parents=True)
    (source / "__pycache__").mkdir()
    (source / "__init__.py").write_text('"""Frozen package."""\n')
    (source / "licenses/NOTICE.txt").write_text("required notice")
    (source / "__pycache__/module.pyc").write_bytes(b"cached bytecode")
    (source / "orphan.pyc").write_bytes(b"orphan bytecode")
    implementation = tmp_path / "implementation"

    manifest = freeze_implementation(implementation, source=source)

    assert manifest == {
        "__init__.py": hashlib.sha256((source / "__init__.py").read_bytes()).hexdigest()
    }
    frozen = implementation / "supermimic"
    assert (frozen / "licenses/NOTICE.txt").read_text() == "required notice"
    assert not (frozen / "__pycache__").exists()
    assert not (frozen / "orphan.pyc").exists()


def test_isolated_command_runs_selected_source_and_preserves_literal_arguments(tmp_path):
    implementation = tmp_path / "implementation with spaces"
    package = implementation / "supermimic"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        "import json\n"
        "def main(arguments):\n"
        "    print(json.dumps({'file': __file__, 'arguments': arguments}))\n"
        "    return 7\n"
    )
    pollution = tmp_path / "ambient"
    (pollution / "supermimic").mkdir(parents=True)
    (pollution / "supermimic/__init__.py").write_text("raise RuntimeError('ambient package')\n")
    environment = {**os.environ, "PYTHONPATH": str(pollution), "PYTHONHOME": str(pollution)}
    arguments = ["export", "--output", "literal $HOME; $(false) path"]

    executed = subprocess.run(
        isolated_cli_command(implementation, arguments),
        env=environment,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert executed.returncode == 7, executed.stderr
    result = json.loads(executed.stdout)
    assert Path(result["file"]) == package / "cli.py"
    assert result["arguments"] == arguments


def test_owned_worker_returns_real_exit_status():
    assert run_owned_worker([sys.executable, "-I", "-c", "raise SystemExit(7)"]) == 7


@pytest.mark.parametrize("timeout", [None, 2.0])
def test_owned_descendant_cannot_survive_its_parent_exit_or_timeout(tmp_path, timeout):
    import signal

    import psutil

    ready = tmp_path / "descendant-pid"
    descendant = (
        "import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        "subprocess.Popen([sys.executable, '-I', '-c', sys.argv[1], sys.argv[2]]); "
        "ready=Path(sys.argv[2]); deadline=time.monotonic()+5\n"
        "while not ready.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "assert ready.exists(), 'Descendant never became ready'\n"
        "time.sleep(60) if sys.argv[3]=='timeout' else None\n"
        "raise SystemExit(7)"
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        parent,
        descendant,
        str(ready),
        "timeout" if timeout else "exit",
    ]
    process = None
    try:
        if timeout:
            with pytest.raises(subprocess.TimeoutExpired):
                run_owned_worker(command, timeout=timeout, grace_seconds=0.05)
        else:
            assert run_owned_worker(command, grace_seconds=0.05) == 7
        assert ready.exists()
        try:
            process = psutil.Process(int(ready.read_text()))
        except psutil.NoSuchProcess:
            return
        deadline = time.monotonic() + 2
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "Owned descendant survived cleanup"
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    finally:
        if process is None and ready.exists():
            try:
                process = psutil.Process(int(ready.read_text()))
            except psutil.NoSuchProcess:
                pass
        if process is not None and process.is_running():
            try:
                process.send_signal(signal.SIGKILL)
            except psutil.NoSuchProcess:
                pass


def test_parent_interruption_reaps_an_owned_worker_that_ignores_sigint(tmp_path, monkeypatch):
    import signal

    import supermimic.runtime as runtime

    ready = tmp_path / "ready"
    command = [
        sys.executable,
        "-I",
        "-c",
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text('ready'); time.sleep(60)",
        str(ready),
    ]
    real_popen = subprocess.Popen
    children = []

    class InterruptedWait:
        def __init__(self, child):
            self.child = child
            self.pid = child.pid
            self.first = True

        def poll(self):
            return self.child.poll()

        def wait(self, **kwargs):
            if self.first:
                self.first = False
                until = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < until:
                    time.sleep(0.01)
                if not ready.exists():
                    raise RuntimeError("Worker never became ready")
                raise KeyboardInterrupt
            return self.child.wait(**kwargs)

    def create(arguments, *, start_new_session, stdout, stderr):
        child = real_popen(
            arguments, start_new_session=start_new_session, stdout=stdout, stderr=stderr
        )
        children.append(child)
        return InterruptedWait(child)

    monkeypatch.setattr(runtime.subprocess, "Popen", create)
    with pytest.raises(KeyboardInterrupt):
        run_owned_worker(command, grace_seconds=0.05)
    assert len(children) == 1 and children[0].returncode == -signal.SIGKILL
