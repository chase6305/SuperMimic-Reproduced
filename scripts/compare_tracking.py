"""Compare uninterrupted physical trajectories against the same reference clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.labels) != len(args.runs):
        parser.error("Each run needs a label")
    from supermimic.cpu_resources import cpu_guard

    with cpu_guard(args.output):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from supermimic.artifacts import sha256, write_json
        from supermimic.deployment.core import heading_error

        motion_hash = sha256(args.motion)
        with np.load(args.motion, allow_pickle=False) as data:
            root = data["body_names"].tolist().index("pelvis")
            position = data["body_pos_w"][:, root].copy()
            orientation = data["body_quat_w"][:, root].copy()
            joints = data["joint_pos"].copy()
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        axes[0, 0].plot(position[:, 0], position[:, 1], color="black", label="reference", alpha=0.6)
        report = {
            "motion_sha256": motion_hash,
            "source": "recorded actual physical states",
            "runs": [],
        }
        for label, run in zip(args.labels, args.runs, strict=True):
            result = json.loads((run / "result.json").read_text())
            trajectory = run / "trajectory.npz"
            digest = sha256(trajectory)
            if (
                result["status"] != "complete"
                or result["auto_reset"]
                or result["motion_sha256"] != motion_hash
                or result["trajectory_sha256"] != digest
            ):
                raise ValueError(f"Incompatible or modified deployment: {run}")
            with np.load(trajectory, allow_pickle=False) as data:
                states, frames = data["states"], data["reference_frame"]
                dt = float(data["control_dt"])
            expected = np.minimum(
                np.arange(len(frames)) + result["reference_goal_offset"], len(joints) - 1
            )
            if not np.array_equal(frames, expected) or len(states) != len(frames) + 1:
                raise ValueError(f"Trajectory clock discontinuity: {run}")
            time = np.arange(1, len(frames) + 1) * dt
            xy = np.linalg.norm(states[1:, :2] - position[frames, :2], axis=-1)
            yaw = heading_error(states[1:, 3:7], orientation[frames])
            joint = np.sqrt(np.mean((states[1:, 7:36] - joints[frames]) ** 2, axis=-1))
            axes[0, 0].plot(states[:, 0], states[:, 1], label=label, alpha=0.8)
            axes[0, 1].plot(time, xy, label=label, linewidth=0.9)
            axes[1, 0].plot(time, np.rad2deg(np.abs(yaw)), label=label, linewidth=0.9)
            axes[1, 1].plot(time, joint, label=label, linewidth=0.7, alpha=0.8)
            report["runs"].append(
                {
                    "label": label,
                    "run": str(run.resolve()),
                    "trajectory_sha256": digest,
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "completed_motion": result["completed_motion"],
                    "simulated_seconds": result["simulated_seconds"],
                    "root_xy_rmse_m": float(np.sqrt(np.mean(xy**2))),
                    "root_yaw_rmse_rad": float(np.sqrt(np.mean(yaw**2))),
                    "joint_rmse_rad": float(np.sqrt(np.mean(joint**2))),
                    "full_clip_return": result["tracking"][
                        "normalized_return_full_clip_zero_padded"
                    ],
                }
            )
        axes[0, 0].set(title="Full XY path", xlabel="x (m)", ylabel="y (m)")
        axes[0, 0].set_aspect("equal", adjustable="datalim")
        axes[0, 1].set(title="Root path error", xlabel="time (s)", ylabel="m")
        axes[1, 0].set(title="Absolute wrapped heading error", xlabel="time (s)", ylabel="degrees")
        axes[1, 1].set(title="Joint RMSE per control", xlabel="time (s)", ylabel="rad")
        for ax in axes.flat:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(
            "G1 uninterrupted CPU MuJoCo: original reference and actual policy trajectories"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
        write_json(args.output.with_suffix(".json"), report)
    print(args.output)


if __name__ == "__main__":
    main()
