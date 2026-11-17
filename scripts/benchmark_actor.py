"""Time complete actor updates on fixed motion states, without physical simulation.

Run under the shared heavy-work lock. Each invocation supervises one CUDA worker,
freezes its implementation and records cold-start time separately from warmed
updates. Compare runs from separate checkouts with the same motion and options.
This measures BPTT and critic fitting, not physical training throughput or quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compact", action="store_true")
    p.add_argument(
        "--compile-policy",
        nargs="?",
        const="default",
        default=False,
        choices=("default", "reduce-overhead"),
    )
    p.add_argument("--batch-size", type=int, choices=(32, 64), default=64)
    p.add_argument("--horizon", type=int, choices=(8, 32), default=32)
    p.add_argument("--repeats", type=int, choices=range(1, 16), default=7)
    p.add_argument("--seed", type=int, default=824)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def worker(args: argparse.Namespace) -> None:
    import hashlib
    import importlib.metadata
    import platform
    import resource
    import statistics
    import time
    from dataclasses import asdict
    from types import SimpleNamespace

    import numpy as np
    import torch

    from supermimic.artifacts import sha256, snapshot_file, write_json
    from supermimic.data.motion_convert import g1_xml_path, load_g1_model
    from supermimic.data.motion_library import MotionLibrary
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.imagination.policy_trainer import PolicyTrainer
    from supermimic.interfaces import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.policy.nets import Critic, Policy
    from supermimic.profiles import training_config
    from supermimic.resources import ResourceLimits, constrain_torch
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel

    constrain_torch(ResourceLimits())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = "cuda:0"  # The parent maps the selected physical device to local index zero.
    cfg = training_config("g1", args.motion.parent, args.output, compile_policy=args.compile_policy)
    cfg.policy_trainer.batch_size = args.batch_size
    cfg.policy_trainer.horizon = args.horizon
    cfg.policy_trainer.obs_noise = True
    model = load_g1_model()
    q0 = torch.tensor(model.qpos0[7:], device=device, dtype=torch.float32)
    motion_digest = snapshot_file(args.motion, args.output / "input.npz")
    lib = MotionLibrary(
        [args.output / "input.npz"], TRACKED_BODIES, q0, device, compact=args.compact
    )
    history = cfg.policy.history
    available = lib.num_frames - history - args.horizon - cfg.env.future_frames
    if available < 1:
        raise ValueError("Motion is too short for the requested history and horizon")
    limits = model.jnt_range[1:]
    center, half = limits.mean(axis=1), (limits[:, 1] - limits[:, 0]) * 0.45
    soft = torch.tensor(
        np.stack([center - half, center + half], axis=-1), device=device, dtype=torch.float32
    )
    heads = AnalyticHeads(
        G1Kinematics(model, device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, soft, q0
    )
    wm = WorldModel(cfg.world_model, heads, device)
    with torch.no_grad():
        # Nonzero Jacobians exercise all dynamics layers, including feedback to the actor.
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.001)
    policy = Policy(cfg.policy).to(device)
    critic = Critic(cfg.critic, pcfg=cfg.policy).to(device)
    frames = (torch.arange(args.batch_size, device=device) * 10 + 50) % available
    states = torch.stack([lib.ref71(frames + t) for t in range(history)])
    states[..., 39:42] = quat_apply_inverse(states[..., 3:7], states[..., 39:42])
    actions = torch.zeros(history, args.batch_size, 29, device=device)
    xi = torch.zeros(args.batch_size, 33, device=device)
    xi[:, 0] = 0.75
    start = StartBatch(states[-1], xi, frames + history - 1, actions[-1], states, actions)
    buffer = SimpleNamespace(sample_starts=lambda *_: start)
    trainer = PolicyTrainer(wm, policy, critic, lib, cfg.policy_trainer)

    def digest(module):
        value = hashlib.sha256()
        for key, tensor in module.state_dict().items():
            value.update(key.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    report = {
        "scope": "PolicyTrainer.update on fixed replay starts: BPTT, clipping, Adam and critic fitting; excludes physics, replay sampling and world-model fitting",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "mujoco_version": importlib.metadata.version("mujoco"),
        "mjlab_version": importlib.metadata.version("mjlab"),
        "g1_xml_sha256": sha256(g1_xml_path()),
        "torch_threads": torch.get_num_threads(),
        "device": torch.cuda.get_device_name(),
        "tf32": False,
        "motion_sha256": motion_digest,
        "benchmark_script_sha256": sha256(__file__),
        "seed": args.seed,
        "compact": args.compact,
        "start_frames": frames.cpu().tolist(),
        "world_model_config": asdict(cfg.world_model),
        "policy_config": asdict(cfg.policy),
        "trainer_config": asdict(cfg.policy_trainer),
        "initial_policy_sha256": digest(policy),
        "initial_world_sha256": digest(wm),
        "initial_critic_sha256": digest(critic),
        "warmup_updates": 2,
        "updates_per_sample": 3,
    }
    torch.manual_seed(args.seed + 23)
    begun = time.perf_counter()
    report["first_update_metrics"] = trainer.update(buffer)
    torch.cuda.synchronize()
    report["first_update_elapsed_s"] = time.perf_counter() - begun
    torch.save(
        {
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "policy_gradients": {key: p.grad for key, p in policy.named_parameters()},
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        args.output / "first-update.pt",
    )
    begun = time.perf_counter()
    for _ in range(2):
        trainer.update(buffer)
    torch.cuda.synchronize()
    report["warmup_updates_elapsed_s"] = time.perf_counter() - begun
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        begun = time.perf_counter()
        for _ in range(3):
            metrics = trainer.update(buffer)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begun) * 1000 / 3)
    report.update(
        median_ms=statistics.median(samples),
        samples_ms=samples,
        peak_torch_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        final_metrics=metrics,
    )
    write_json(args.output / "benchmark.json", report)
    print(json.dumps({key: report[key] for key in ("median_ms", "first_update_elapsed_s")}))


def main() -> int:
    args = parser().parse_args()
    args.motion = args.motion.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.worker:
        worker(args)
        return 0
    if not args.motion.is_file():
        raise FileNotFoundError(args.motion)
    if args.motion.suffix.lower() != ".npz":
        raise ValueError("--motion requires one NPZ clip")

    # A second checkout may reuse the same interpreter; benchmark that checkout's
    # package rather than an editable installation pointing at another branch.
    checkout_source = Path(__file__).resolve().parents[1] / "src"
    if (checkout_source / "supermimic" / "__init__.py").is_file():
        sys.path.insert(0, str(checkout_source))

    from supermimic.resources import GIB, ResourceLimits, host_snapshot
    from supermimic.supervisor import supervise

    # Honor the shared-workstation reserve as well as the per-worker budgets.
    existing = args.output.parent
    while not existing.exists():
        existing = existing.parent
    reserve = max(12.0, host_snapshot(existing)["ram_total_bytes"] * 0.2 / GIB)
    limits = ResourceLimits(gpu_index=args.gpu_index, reserve_ram_gib=reserve)
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
        limits,
        max_seconds=600,
        freeze_source=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
