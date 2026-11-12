"""Pack filtered npz clips into one compact library file (fp16 71-d reference states).

uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv --out data/library/bones_seed_compact.pt
"""

from __future__ import annotations

from pathlib import Path

import tyro


def _pack(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 2) -> None:
    import os
    import zipfile

    import numpy as np
    import torch
    from tqdm import tqdm

    from supermimic.data.discovery import discover_npz
    from supermimic.data.schema import validate_motion
    from supermimic.resources import GIB, host_snapshot

    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four CPU packing workers")
    if out.exists():
        raise FileExistsError(out)
    files = discover_npz(npz_dir, manifest=manifest)
    if not files:
        raise ValueError("No motion clips selected")
    unpacked = 0
    for path in files:
        with zipfile.ZipFile(path) as archive:
            unpacked += sum(info.file_size for info in archive.infolist())
    if unpacked > 2 * GIB or host_snapshot()["ram_available_bytes"] < 8 * GIB + unpacked * 3:
        raise RuntimeError("Packing would exceed the 2 GiB input or 8 GiB host RAM reserve")
    print(f"{len(files)} clips")
    from concurrent.futures import ThreadPoolExecutor

    def load(f):
        validate_motion(f)
        with np.load(f) as d:
            names = [str(n) for n in d["body_names"]]
            r = names.index("pelvis")
            ref = np.concatenate(
                [
                    d["body_pos_w"][:, r],
                    d["body_quat_w"][:, r],
                    d["joint_pos"],
                    d["body_lin_vel_w"][:, r],
                    d["body_ang_vel_w"][:, r],
                    d["joint_vel"],
                ],
                axis=1,
            )
        return f.stem, ref.astype(np.float16)

    with ThreadPoolExecutor(workers) as ex:
        items = list(tqdm(ex.map(load, files), total=len(files)))
    names = [n for n, _ in items]
    lengths = [int(r.shape[0]) for _, r in items]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ref = torch.from_numpy(np.concatenate([r for _, r in items], 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name("." + out.name + ".tmp")
    try:
        torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths}, temporary)
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"saved {out}: {ref.shape[0]} frames ({ref.shape[0] / 50 / 3600:.2f} h), {ref.numel() * 2 / 1e9:.2f} GB fp16"
    )


def main(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 2) -> None:
    import os

    from supermimic.cpu_resources import cpu_guard

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with cpu_guard(out):
        _pack(npz_dir, out, manifest, workers)


if __name__ == "__main__":
    tyro.cli(main)
