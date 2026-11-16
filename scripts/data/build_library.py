"""Pack filtered npz clips into one compact library file (fp16 71-d reference states).

uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv --out data/library/bones_seed_compact.pt
"""

from __future__ import annotations

from pathlib import Path

import tyro


def _pack(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 2) -> None:
    from tqdm import tqdm

    from supermimic.data.discovery import discover_npz
    from supermimic.data.packing import pack_npz_motions

    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four CPU packing workers")
    if out.exists():
        raise FileExistsError(out)
    files = discover_npz(npz_dir, manifest=manifest)
    if not files:
        raise ValueError("No motion clips selected")
    print(f"{len(files)} clips")
    with tqdm(total=len(files)) as progress:
        ref = pack_npz_motions(files, out, workers=workers, progress=progress.update).ref
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
