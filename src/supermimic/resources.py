"""Resource accounting and conservative admission for a shared workstation.

The GPU counter includes allocations by Warp, MuJoCo and Torch. Torch's allocator
limit alone cannot bound the simulator. No other process is signalled or changed.
"""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

MIB = 1024**2
GIB = 1024**3


class ResourceUnavailable(RuntimeError):
    """A temporary admission failure; callers may wait without changing other tasks."""


@dataclass
class ResourceLimits:
    gpu_index: int = 0
    max_gpu_mib: int = 2304
    reserve_gpu_mib: int = 4096
    max_rss_gib: float = 6.0
    reserve_ram_gib: float = 8.0
    reserve_disk_gib: float = 20.0
    poll_seconds: float = 2.0
    stop_grace_seconds: float = 30.0

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < (0 if key == "gpu_index" else 0.001):
                raise ValueError(f"Invalid resource limit {key}={value}")


def gpu_snapshot() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        dict(
            index=int(row[0]),
            uuid=row[1].strip(),
            total_mib=int(row[2]),
            used_mib=int(row[3]),
            free_mib=int(row[4]),
        )
        for row in csv.reader(io.StringIO(result.stdout))
        if len(row) == 5
    ]


def process_gpu_mib(pids: set[int], gpu_uuid: str) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        return sum(
            int(row[2])
            for row in csv.reader(io.StringIO(result.stdout))
            if len(row) == 3 and int(row[0]) in pids and row[1].strip() == gpu_uuid
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def host_snapshot(path: Path | str = ".") -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            mem[key] = int(value.split()[0]) * 1024
    usage = shutil.disk_usage(path)
    return {
        "ram_available_bytes": mem["MemAvailable"],
        "ram_total_bytes": mem["MemTotal"],
        "swap_free_bytes": mem["SwapFree"],
        "disk_free_bytes": usage.free,
    }


def admit(limits: ResourceLimits, path: Path | str = ".", *, cuda: bool = True) -> dict:
    limits.validate()
    host = host_snapshot(path)
    if host["ram_available_bytes"] < (limits.reserve_ram_gib + limits.max_rss_gib) * GIB:
        raise ResourceUnavailable(
            "Insufficient available RAM for the process budget plus system reserve"
        )
    if host["disk_free_bytes"] < limits.reserve_disk_gib * GIB:
        raise ResourceUnavailable("Disk free space is below the configured reserve")
    gpu = next((g for g in gpu_snapshot() if g["index"] == limits.gpu_index), None)
    if cuda and gpu is None:
        raise ResourceUnavailable("Requested NVIDIA GPU is unavailable")
    if cuda and gpu is not None and gpu["free_mib"] < limits.max_gpu_mib + limits.reserve_gpu_mib:
        raise ResourceUnavailable(
            f"GPU {limits.gpu_index} has {gpu['free_mib']} MiB free; "
            f"need {limits.max_gpu_mib} MiB budget + {limits.reserve_gpu_mib} MiB reserve"
        )
    return {"host": host, "gpu": gpu, "limits": asdict(limits)}


def constrain_torch(limits: ResourceLimits) -> None:
    """Reserve room inside the process budget for non-Torch CUDA allocations."""
    import torch

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mjlab training")
    total = torch.cuda.get_device_properties(0).total_memory
    # Simulator/context allocations are monitored separately by the parent process.
    torch_bytes = max(256, limits.max_gpu_mib - 1024) * MIB
    torch.cuda.set_per_process_memory_fraction(min(torch_bytes / total, 0.8), 0)


def clean_environment() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "MASTER_ADDR", "MASTER_PORT"}
    }
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        CTEST_PARALLEL_LEVEL="1",
        MAX_JOBS="2",
        RAYON_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_WORLD_SIZE="1",
        LOCAL_RANK="0",
    )
    return env
