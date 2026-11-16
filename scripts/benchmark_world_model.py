"""Measure a fixed world-model window and parameter backward on CUDA.

Run under the shared heavy-work lock. Each run freezes its package and this
script, uses a private compiler cache, and reports first-call time separately
from warmed samples. This excludes replay sampling, Adam and physical training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("eager", "step", "window"), default="window")
    p.add_argument("--width", type=int, choices=(512, 2048), default=512)
    p.add_argument("--batch-size", type=int, choices=(128, 512), default=128)
    p.add_argument("--window", type=int, choices=(1, 8, 12, 64), default=8)
    p.add_argument("--repeats", type=int, choices=range(1, 16), default=7)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def worker(args: argparse.Namespace) -> None:
    import hashlib
    import json
    import os
    import platform
    import statistics
    import time
    from dataclasses import asdict
    from types import SimpleNamespace

    # Configure these before importing Torch or its compiler. Driver caches may
    # still persist between processes; this isolates the Torch/Triton disk caches.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(args.output / "triton-cache")
    import torch

    from supermimic.artifacts import sha256, write_json
    from supermimic.config import WorldModelCfg
    from supermimic.resources import ResourceLimits, constrain_torch
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.reference import ReferenceFrame
    from supermimic.world_model.world_model import WorldModel

    constrain_torch(ResourceLimits())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    cfg = WorldModelCfg(hidden=(args.width,) * 3)
    wm = WorldModel(
        cfg, SimpleNamespace(termination_features=AnalyticHeads.termination_features), "cuda:0"
    )
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.001)
    k, n = args.window, args.batch_size
    states = torch.randn(k + 1, n, 71, device="cuda:0") * 0.05
    states[..., 2] += 0.78
    states[..., 3] += 1
    states[..., 3:7] /= states[..., 3:7].norm(dim=-1, keepdim=True)
    actions = torch.randn(k, n, 29, device="cuda:0") * 0.1
    context = torch.randn(n, 33, device="cuda:0") * 0.001
    context[:, 0] += 0.75
    done = (torch.rand(k, n, device="cuda:0") < 0.1).float()
    reference_states = states[1:].clone()
    refs = [
        ReferenceFrame(None, None, None, None, s[..., :3], s[..., 3:7]) for s in reference_states
    ]
    wm.enable_compile(None if args.backend == "eager" else "default")
    if args.backend == "step":
        wm._MAX_COMPILED_WINDOW = 0

    def digest(values):
        result = hashlib.sha256()
        for value in values:
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    report = {
        "scope": "Fixed synthetic window loss and parameter backward; excludes replay sampling, Adam and physical simulation",
        "backend": args.backend,
        "seed": args.seed,
        "world_model_config": asdict(cfg),
        "batch_size": n,
        "window": k,
        "reference_storage": "independent clone of logged target states",
        "initial_weights_sha256": digest(wm.state_dict().values()),
        "inputs_sha256": digest([states, actions, context, done, reference_states]),
        "benchmark_script_sha256": sha256(__file__),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "tf32": False,
        "torch_threads": torch.get_num_threads(),
        "private_torch_triton_caches": True,
        "warmup_calls": 3,
        "operations_per_sample": 20,
    }
    started = time.perf_counter()
    loss, metrics = wm.window_loss(states, actions, context, done, refs)
    gradients = torch.autograd.grad(loss, tuple(wm.parameters()))
    torch.cuda.synchronize()
    report["first_forward_backward_s"] = time.perf_counter() - started
    report["first_loss"] = float(loss.detach())
    torch.save(
        {"loss": loss.detach(), "metrics": metrics, "gradients": gradients},
        args.output / "first.pt",
    )
    del loss, metrics, gradients

    def forward_backward():
        wm.zero_grad(set_to_none=True)
        loss, _ = wm.window_loss(states, actions, context, done, refs)
        loss.backward()

    for _ in range(3):
        forward_backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        for _ in range(20):
            forward_backward()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000 / 20)
    report.update(
        median_ms=statistics.median(samples),
        samples_ms=samples,
        peak_torch_bytes=torch.cuda.max_memory_allocated(),
    )
    write_json(args.output / "benchmark.json", report)
    print(json.dumps({key: report[key] for key in ("median_ms", "first_forward_backward_s")}))


def main() -> int:
    args = parser().parse_args()
    args.output = args.output.expanduser().resolve()
    if args.backend == "window" and args.window > 8:
        parser().error("window compilation is bounded to at most eight steps; use step or eager")
    if args.worker:
        worker(args)
        return 0
    source = Path(__file__).resolve().parents[1] / "src"
    if (source / "supermimic" / "__init__.py").is_file():
        sys.path.insert(0, str(source))

    from supermimic.resources import GIB, ResourceLimits, host_snapshot
    from supermimic.supervisor import supervise

    existing = args.output.parent
    while not existing.exists():
        existing = existing.parent
    reserve = max(12.0, host_snapshot(existing)["ram_total_bytes"] * 0.2 / GIB)
    bootstrap = (
        "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
        "from supermimic.artifacts import snapshot_file; "
        "script=sys.argv.pop(1); snapshot=sys.argv.pop(1); "
        "snapshot_file(script,snapshot); runpy.run_path(snapshot,run_name='__main__')"
    )
    return supervise(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(args.output / "implementation"),
            str(Path(__file__).resolve()),
            str(args.output / "benchmark-script.py"),
            *sys.argv[1:],
            "--worker",
        ],
        args.output,
        ResourceLimits(gpu_index=args.gpu_index, reserve_ram_gib=reserve),
        max_seconds=600,
        freeze_source=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
