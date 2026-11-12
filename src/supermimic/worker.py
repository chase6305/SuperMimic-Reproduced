"""Isolated worker for guarded train/eval/play operations."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from supermimic.artifacts import sha256, snapshot_file, write_json
from supermimic.resources import ResourceLimits, constrain_torch


def run_worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    options = config["options"]
    output = Path(options["output"])
    write_json(output / "options.json", config)
    started = time.monotonic()
    record = dict(status="preparing", official_weights_used=False, paper_scale_result=False)
    trainer = None
    try:
        from supermimic.data.discovery import discover_npz
        from supermimic.data.schema import validate_motion

        files = discover_npz(options["motions"], names=options.get("clips"))
        if not files:
            raise FileNotFoundError("No selected G1 motions")
        if options.get("clips") and {f.stem for f in files} != set(options["clips"]):
            raise ValueError("Some requested clips are missing")
        motion_files = {}
        private_motions = output / "inputs/motions"
        private_motions.mkdir(parents=True, exist_ok=False)
        snapshots = []
        manifest_path = Path(options["motions"]) / "manifest.json"
        expected_motions = None
        if manifest_path.is_file():
            prepared = json.loads(manifest_path.read_text())
            if prepared.get("robot") != "unitree-g1-29dof":
                raise ValueError("Prepared motion manifest has an incompatible robot")
            expected_motions = prepared["files"]
        for path in files:
            if expected_motions is not None and path.name not in expected_motions:
                raise ValueError(f"Motion differs from the prepared manifest: {path}")
            # Reject invalid/unbounded archives before copying them into the run.
            # Validate again below so recorded metadata belongs to the pinned copy.
            validate_motion(path)
            dest = private_motions / path.name
            try:
                digest = snapshot_file(
                    path,
                    dest,
                    expected_sha256=(
                        expected_motions[path.name]["sha256"]
                        if expected_motions is not None
                        else None
                    ),
                )
            except ValueError as error:
                raise ValueError(f"Motion differs from the prepared manifest: {path}") from error
            except RuntimeError as error:
                raise RuntimeError(f"Motion changed during snapshot: {path}") from error
            info = validate_motion(dest)
            snapshots.append(dest)
            motion_files[path.name] = dict(
                sha256=digest, frames=info["frames"], source=str(path.resolve())
            )
        files = snapshots
        write_json(
            private_motions / "manifest.json",
            {"robot": "unitree-g1-29dof", "quaternion_order": "wxyz", "files": motion_files},
        )
        record["motion_files"] = motion_files
        record["input_motion_snapshot"] = str(private_motions)
        write_json(output / "result.json", record)
        constrain_torch(ResourceLimits(**config["limits"]))
        checkpoint = Path(options["checkpoint"]) if options.get("checkpoint") else None
        if checkpoint:
            private_checkpoint = output / "inputs/checkpoint.pt"
            try:
                digest = snapshot_file(checkpoint, private_checkpoint)
            except RuntimeError as error:
                raise RuntimeError("Checkpoint changed during snapshot") from error
            record["input_checkpoint_sha256"] = digest
            checkpoint = private_checkpoint
        if options["command"] == "train":
            from supermimic.memory import validate_training_budget
            from supermimic.profiles import training_config
            from supermimic.training.loop import Trainer

            cfg = training_config(
                options["profile"],
                private_motions,
                output,
                num_envs=options.get("num_envs"),
                iterations=options["iterations"],
                hours=options["hours"],
                checkpoint_interval=options.get("checkpoint_interval"),
                seed=options["seed"],
                clips=options.get("clips"),
                resume=checkpoint,
                compile_policy=options.get("compile_policy", False),
                compile_world_model=options.get("compile_world_model", False),
                wm_batches_per_update=options.get("wm_batches_per_update"),
                policy_batch_size=options.get("policy_batch_size"),
                policy_horizon=options.get("policy_horizon"),
                policy_update_scope=options.get("policy_update_scope"),
                wm_batch_size=options.get("wm_batch_size"),
                replay_capacity=options.get("replay_capacity"),
                transfer=options.get("transfer", False),
                action_clip=options.get("action_clip"),
                tracking_objective=options.get("tracking_objective"),
                training_mode=options.get("training_mode"),
                learning_rate=options.get("learning_rate"),
                wm_one_step_batches=options.get("wm_one_step_batches"),
                completion_aware_sampling=options.get("completion_aware_sampling"),
                foot_support_velocity_weight=options.get("foot_support_velocity_weight"),
                effector_position_weight=options.get("effector_position_weight"),
                effector_position_weights=options.get("effector_position_weights"),
                joint_tracking_weight=options.get("joint_tracking_weight"),
                root_xy_tracking_weight=options.get("root_xy_tracking_weight"),
                use_critic=options.get("critic"),
                policy_gradient_accumulation=options.get("policy_gradient_accumulation"),
                wm_window=options.get("wm_window"),
                priority_windows_s=options.get("priority_window"),
                priority_reset_ratio=options.get("priority_reset_ratio"),
            )
            write_json(output / "config.json", asdict(cfg))
            record["memory_estimate"] = validate_training_budget(
                cfg,
                sum(info["frames"] for info in motion_files.values()),
                config["limits"]["max_gpu_mib"],
            )
            write_json(output / "result.json", record)
            trainer = Trainer(cfg)
            try:
                record.update(
                    status="running",
                    starting_iteration=trainer.iteration,
                    starting_updates=trainer.num_updates,
                    transfer_provenance=trainer.transfer_provenance,
                )
                write_json(output / "result.json", record)
                trainer.run()
                record.update(
                    iteration=trainer.iteration,
                    updates=trainer.num_updates,
                    stop_reason=trainer.stop_reason,
                    checkpoint=str(trainer.log_dir / "ckpt_latest.pt"),
                    checkpoint_sha256=sha256(trainer.log_dir / "ckpt_latest.pt"),
                )
            finally:
                # run() releases resources too; close() is safe to repeat and also
                # covers failures before run() installs its own cleanup handler.
                trainer.close()
        elif options["command"] == "eval":
            from supermimic.eval.benchmark import (
                load_policy_from_checkpoint,
                run_benchmark,
                save_results,
                summarize,
            )

            policy, cfg = load_policy_from_checkpoint(checkpoint, "cuda:0")
            results = run_benchmark(
                policy,
                [str(f) for f in files],
                num_envs=options.get("num_envs") or 1,
                future_frames=cfg["env"]["future_frames"],
                anchor_obs=cfg["env"].get("anchor_obs", False),
            )
            save_results(results, output / "evaluation.json")
            record["evaluation"] = summarize(results)
        else:
            from supermimic.deployment.play import play_checkpoint

            play_checkpoint(checkpoint, files, options.get("num_envs") or 1, options["viewer"])
        # Input mutation is a failed run, even if computation completed normally.
        for path in files:
            if sha256(path) != motion_files[path.name]["sha256"]:
                raise RuntimeError(f"Motion input changed during the run: {path}")
        record["status"] = "complete"
        return 0
    except BaseException as error:  # noqa: BLE001 - isolated worker must record cancellation too
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        if trainer is not None:
            record.update(iteration=trainer.iteration, updates=trainer.num_updates)
            latest = trainer.log_dir / "ckpt_latest.pt"
            if latest.is_file():
                record["recoverable_checkpoint"] = str(latest)
                record["recoverable_checkpoint_sha256"] = sha256(latest)
        traceback.print_exc()
        return 1
    finally:
        record["elapsed_s"] = time.monotonic() - started
        write_json(output / "result.json", record)
