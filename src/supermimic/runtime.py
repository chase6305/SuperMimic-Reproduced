"""Lightweight source snapshots and isolated entrypoints for supervised workers."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from supermimic.artifacts import sha256, write_json

_CLI_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from supermimic.cli import main; raise SystemExit(main(sys.argv[2:]))"
)


def source_manifest(package: Path | None = None) -> dict[str, str]:
    """Hash all package Python sources, not an exact set of imported modules."""
    package = package if package is not None else Path(__file__).parent
    return {str(path.relative_to(package)): sha256(path) for path in sorted(package.rglob("*.py"))}


def freeze_implementation(implementation: Path, *, source: Path | None = None) -> dict[str, str]:
    """Copy the package and its assets; return Python hashes relative to the package."""
    package = source if source is not None else Path(__file__).parent
    frozen = implementation / "supermimic"
    shutil.copytree(package, frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return source_manifest(frozen)


def isolated_cli_command(implementation: Path, arguments: Sequence[str]) -> list[str]:
    """Run CLI arguments with this interpreter and only the selected package override."""
    return [sys.executable, "-I", "-c", _CLI_BOOTSTRAP, str(implementation), *arguments]


@contextmanager
def temporary_worker_config(output: Path, payload: dict) -> Iterator[Path]:
    """Own a unique adjacent worker configuration until the context exits."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix="." + output.name + "-", suffix=".worker.json", dir=output.parent
    )
    config_path = Path(name)
    try:
        os.close(fd)
        write_json(config_path, payload)
        yield config_path
    finally:
        config_path.unlink(missing_ok=True)
