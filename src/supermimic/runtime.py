"""Lightweight source snapshots and isolated entrypoints for supervised workers."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from supermimic.artifacts import sha256

_CLI_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from supermimic.cli import main; raise SystemExit(main(sys.argv[2:]))"
)


def freeze_implementation(implementation: Path, *, source: Path | None = None) -> dict[str, str]:
    """Copy the package and its assets; return Python hashes relative to the package."""
    package = source if source is not None else Path(__file__).parent
    frozen = implementation / "supermimic"
    shutil.copytree(package, frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {str(path.relative_to(frozen)): sha256(path) for path in sorted(frozen.rglob("*.py"))}


def isolated_cli_command(implementation: Path, arguments: Sequence[str]) -> list[str]:
    """Run CLI arguments with this interpreter and only the selected package override."""
    return [sys.executable, "-I", "-c", _CLI_BOOTSTRAP, str(implementation), *arguments]
