"""Prepare verified local motion copies without loading numerical SDKs at import."""

from __future__ import annotations

import shutil
from pathlib import Path

from supermimic.artifacts import sha256, write_json


def prepare_motions(source: Path, output: Path, clips: list[str]) -> dict:
    from supermimic.data.schema import validate_motion

    if len(set(clips)) != len(clips):
        raise ValueError("Clip names must be unique")
    sources = []
    for name in clips:
        if Path(name).name != name or name in (".", "..") or name.endswith(".npz"):
            raise ValueError("Use plain clip names without extensions")
        path = source / (name + ".npz")
        # Preflight every source before creating output, retaining the existing
        # validation/hash order. Final metadata is read from the copied bytes.
        validate_motion(path)
        sources.append((path, sha256(path), str(path.resolve())))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(robot="unitree-g1-29dof", quaternion_order="wxyz", files={})
    for path, digest, original_source in sources:
        dest = output / path.name
        shutil.copyfile(path, dest)
        if sha256(dest) != digest:
            raise RuntimeError("Input changed while copying " + str(path))
        info = validate_motion(dest)
        info.update(sha256=digest, source=original_source)
        result["files"][path.name] = info
    write_json(output / "manifest.json", result)
    return result
