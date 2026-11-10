"""Standalone SuperMimic/G1 workflows. Heavy libraries load only inside their operation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from supermimic.artifacts import write_json
from supermimic.resources import ResourceLimits, gpu_snapshot, host_snapshot


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show SDK, GPU and memory availability")
    prepare = sub.add_parser("prepare", help="Copy and validate selected local G1 NPZ clips")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", type=Path, default=Path("data/npz/g1"))
    prepare.add_argument("--clips", nargs="+", required=True)
    for action in ("train", "eval", "play"):
        q = sub.add_parser(action)
        q.add_argument("--motions", type=Path, default=Path("data/npz/g1"))
        q.add_argument("--clips", nargs="+")
        q.add_argument("--checkpoint", type=Path)
        q.add_argument("--output", type=Path)
        q.add_argument("--num-envs", type=positive_int)
        q.add_argument("--hours", type=positive_float, default=1.0)
        q.add_argument(
            "--checkpoint-interval",
            type=positive_int,
            default=None,
            help="Iterations between saved checkpoints; default 500, final head always saved",
        )
        q.add_argument("--max-gpu-mib", type=positive_int, default=2304)
        q.add_argument("--reserve-gpu-mib", type=positive_int, default=4096)
        q.add_argument("--max-rss-gib", type=positive_float, default=6.0)
        q.add_argument("--dry-run", action="store_true")
        if action == "train":
            q.add_argument(
                "--transfer",
                action="store_true",
                help="Initialize network weights on new motions; reset optimizers, sampler and counters",
            )
            q.add_argument("--profile", choices=("smoke", "g1", "paper-width"), default="smoke")
            q.add_argument(
                "--completion-aware-sampling",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Correct successful clip-end returns before choosing difficult reset bins",
            )
            q.add_argument("--iterations", type=positive_int, default=100000)
            q.add_argument("--seed", type=int, default=0)
            q.add_argument(
                "--priority-window",
                action="append",
                nargs=2,
                type=nonnegative_float,
                metavar=("START", "END"),
                help="Increase reset sampling in a difficult time window of a single full clip (seconds)",
            )
            q.add_argument(
                "--priority-reset-ratio",
                type=nonnegative_float,
                help="Priority mixture share, within [0, 0.5]; defaults to 0.25 when windows are specified",
            )
            q.add_argument("--wm-batches-per-update", type=positive_int)
            q.add_argument(
                "--wm-window",
                type=positive_int,
                help="World-model autoregressive fitting window (1-64 controls); saved value is inherited",
            )
            q.add_argument(
                "--wm-one-step-batches",
                type=nonnegative_int,
                help="Supplement long windows with short-episode transitions",
            )
            q.add_argument("--wm-batch-size", type=positive_int)
            q.add_argument("--policy-batch-size", type=positive_int)
            q.add_argument(
                "--policy-update-scope",
                choices=("all", "decoder", "action-head"),
                help="Actor parameters to update; keeps frozen feedback derivatives, saved choice inherited on resume",
            )
            q.add_argument(
                "--policy-horizon",
                type=positive_int,
                help="Imagined controls per actor update (1-64); saved value inherited on resume, full physical evaluation unchanged",
            )
            q.add_argument(
                "--policy-gradient-accumulation",
                type=positive_int,
                help="Average sequential actor microbatches per optimizer step (1-16); saved value is inherited",
            )
            q.add_argument(
                "--joint-tracking-weight",
                type=nonnegative_float,
                help="Override dense joint-pose penalty, including wrist rotations; saved value is inherited",
            )
            q.add_argument(
                "--critic",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Use terminal critic bootstrapping and value fitting; enabled by default, saved choice inherited",
            )
            q.add_argument(
                "--root-xy-tracking-weight",
                type=nonnegative_float,
                help="Optional dense world-XY root-position penalty; zero by default, saved value inherited",
            )
            q.add_argument(
                "--effector-position-weight",
                type=nonnegative_float,
                help="Optional torso-relative position penalty for both feet and hands",
            )
            q.add_argument(
                "--effector-position-weights",
                nargs=4,
                type=positive_float,
                metavar=("LEFT_FOOT", "RIGHT_FOOT", "LEFT_HAND", "RIGHT_HAND"),
                help="Relative limb coefficients for the optional position objective; saved values are inherited",
            )
            q.add_argument(
                "--foot-support-velocity-weight",
                type=nonnegative_float,
                help="Override optional stance-normal-velocity penalty; saved value is inherited",
            )
            q.add_argument("--replay-capacity", type=positive_int)
            q.add_argument(
                "--learning-rate",
                type=positive_float,
                help="Override world-model/actor/critic base learning rate; saved values are inherited on resume",
            )
            q.add_argument(
                "--training-mode",
                choices=("randomized", "deterministic"),
                help="Use deterministic simulation first, or randomized training; saved mode is inherited on resume",
            )
            q.add_argument(
                "--tracking-objective",
                choices=("baseline", "pose", "dynamic", "full-motion", "contact"),
                help="Optional dense imagined imitation objective; evaluation reward unchanged",
            )
            q.add_argument(
                "--action-clip",
                type=positive_float,
                help="Bound normalized policy actions in training and exported actor",
            )
            q.add_argument(
                "--compile-policy",
                action="store_true",
                help="Compile imagined physics steps inside the same resource limits",
            )
            q.add_argument(
                "--compile-world-model",
                action="store_true",
                help="Compile world-model fitting inside the same resource limits",
            )
        if action == "play":
            q.add_argument("--viewer", choices=("native", "viser"), default="viser")
    export = sub.add_parser("export", help="Export a trusted checkpoint and G1 control contract")
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    anchor = sub.add_parser("enable-anchor", help="Warm-start optional relative-pose feedback")
    anchor.add_argument("--checkpoint", required=True, type=Path)
    anchor.add_argument("--output", required=True, type=Path)
    deploy = sub.add_parser("deploy", help="Run an exported policy in CPU MuJoCo")
    deploy.add_argument("--bundle", required=True, type=Path)
    deploy.add_argument("--motion", required=True, type=Path)
    deploy.add_argument("--output", required=True, type=Path)
    deploy.add_argument("--steps", type=positive_int)
    deploy.add_argument("--viewer", action="store_true")
    render = sub.add_parser("render", help="Render saved MuJoCo states on CPU")
    render.add_argument("--bundle", required=True, type=Path)
    render.add_argument("--trajectory", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--seconds", type=positive_float, default=10.0)
    render.add_argument("--start-seconds", type=nonnegative_float, default=0.0)
    render.add_argument("--fps", type=int, choices=(10, 25, 50), default=25)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("config", type=Path)
    return p


def prepare_motions(source: Path, output: Path, clips: list[str]) -> dict:
    import shutil

    from supermimic.artifacts import sha256
    from supermimic.data.schema import validate_motion

    if len(set(clips)) != len(clips):
        raise ValueError("Clip names must be unique")
    sources = []
    for name in clips:
        if Path(name).name != name or name in (".", "..") or name.endswith(".npz"):
            raise ValueError("Use plain clip names without extensions")
        path = source / (name + ".npz")
        info = validate_motion(path)
        info.update(sha256=sha256(path), source=str(path.resolve()))
        sources.append((path, info))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(robot="unitree-g1-29dof", quaternion_order="wxyz", files={})
    for path, info in sources:
        dest = output / path.name
        shutil.copyfile(path, dest)
        if sha256(dest) != info["sha256"]:
            raise RuntimeError("Input changed while copying " + str(path))
        result["files"][path.name] = info
    write_json(output / "manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            import importlib.metadata

            versions = {}
            for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            print(
                json.dumps(
                    dict(
                        python=sys.version,
                        packages=versions,
                        gpu=gpu_snapshot(),
                        host=host_snapshot(),
                        training_executed=False,
                    ),
                    indent=2,
                )
            )
        elif args.command == "prepare":
            from supermimic.cpu_resources import cpu_guard

            with cpu_guard(args.output):
                result = prepare_motions(args.source, args.output, args.clips)
            print(json.dumps(result, indent=2))
        elif args.command == "_worker":
            from supermimic.worker import run_worker

            return run_worker(args.config)
        elif args.command == "enable-anchor":
            from supermimic.cpu_resources import cpu_guard

            with cpu_guard(args.output, limits=ResourceLimits(max_rss_gib=3.0)):
                from supermimic.training.anchor import enable_anchor_feedback

                result = enable_anchor_feedback(args.checkpoint, args.output)
            print(json.dumps(result, indent=2))
        elif args.command == "export":
            from supermimic.cpu_resources import cpu_guard

            with cpu_guard(args.output, limits=ResourceLimits(max_rss_gib=2.0)):
                from supermimic.deployment.export import export_policy

                result = export_policy(args.checkpoint, args.output)
            print(json.dumps(result, indent=2))
        elif args.command == "deploy":
            from supermimic.cpu_resources import cpu_guard

            with cpu_guard(args.output, limits=ResourceLimits(max_rss_gib=2.0)):
                from supermimic.deployment.mujoco import deploy

                result = deploy(args.bundle, args.motion, args.output, args.steps, args.viewer)
            print(json.dumps(result, indent=2))
        elif args.command == "render":
            from supermimic.cpu_resources import cpu_guard

            with cpu_guard(args.output, limits=ResourceLimits(max_rss_gib=2.0)):
                from supermimic.deployment.render import render_rollout

                result = render_rollout(
                    args.bundle,
                    args.trajectory,
                    args.output,
                    args.seconds,
                    args.fps,
                    start_seconds=args.start_seconds,
                )
            print(json.dumps(result, indent=2))
        else:
            if args.command == "train" and args.transfer and args.checkpoint is None:
                raise ValueError("--transfer requires --checkpoint")
            if args.command in ("eval", "play") and args.checkpoint is None:
                raise ValueError("--checkpoint is required for eval/play")
            tag = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            output = (args.output or Path("runs") / (tag + "-" + args.command)).resolve()
            options = vars(args).copy()
            options["output"] = output
            limits = ResourceLimits(
                max_gpu_mib=args.max_gpu_mib,
                reserve_gpu_mib=args.reserve_gpu_mib,
                max_rss_gib=args.max_rss_gib,
            )
            if args.dry_run:
                print(
                    json.dumps(dict(options=options, limits=asdict(limits)), indent=2, default=str)
                )
                return 0
            from supermimic.supervisor import supervise

            # Worker config lives beside the new run, not inside it; supervisor owns directory creation.
            output.parent.mkdir(parents=True, exist_ok=True)
            config_path = output.parent / ("." + output.name + ".worker.json")
            if config_path.exists() or output.exists():
                raise FileExistsError(output)
            write_json(config_path, dict(options=options, limits=asdict(limits)))
            try:
                bootstrap = (
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from supermimic.cli import main; raise SystemExit(main(sys.argv[2:]))"
                )
                code = supervise(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        bootstrap,
                        str(output / "implementation"),
                        "_worker",
                        str(config_path),
                    ],
                    output,
                    limits,
                    max_seconds=args.hours * 3600 + 180,
                    freeze_source=True,
                )
            finally:
                config_path.unlink(missing_ok=True)
            print(f"Run record: {output / 'run.json'}; log: {output / 'worker.log'}")
            return code
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"supermimic: {error}", file=sys.stderr)
        return 2


def train_main() -> int:
    return main(["train", *sys.argv[1:]])
