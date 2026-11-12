"""Select local motion paths without importing array, tensor or simulator runtimes."""

from __future__ import annotations

from pathlib import Path


def discover_npz(
    directory: str | Path,
    limit: int | None = None,
    names: list[str] | None = None,
    manifest: str | Path | None = None,
) -> list[Path]:
    """List npz clips; ``manifest`` (csv with name,keep columns) restricts to keep==1."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if manifest is not None:
        import csv

        with open(manifest) as f:
            keep = {
                row["name"]
                for row in csv.DictReader(f)
                if row.get("keep", "1") in ("1", "True", "true")
            }
        files = [f for f in files if f.stem in keep]
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
    if limit is not None:
        files = files[:limit]
    return files
