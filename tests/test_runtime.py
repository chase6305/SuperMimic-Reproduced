"""Frozen CLI execution retains assets and ignores ambient Python package paths."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from supermimic.runtime import freeze_implementation, isolated_cli_command


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
