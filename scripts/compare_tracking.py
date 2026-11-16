"""Compare uninterrupted physical trajectories against the same reference clip."""

from __future__ import annotations

import argparse
from pathlib import Path


def _comparison_reference(motion: Path):
    """Read the plotting fields and digest together without the training ABI."""
    from supermimic.deployment.evidence import read_motion_evidence

    reference = read_motion_evidence(motion, include_orientation=True)
    data = reference.arrays
    root = data["body_names"].tolist().index("pelvis")
    return (
        reference.sha256,
        data["body_pos_w"][:, root].copy(),
        data["body_quat_w"][:, root].copy(),
        data["joint_pos"].copy(),
    )


def _comparison_run(run: Path, motion_sha256: str, motion_frames: int):
    """Prepare one finished rollout without requiring actions, contacts or success."""
    from supermimic.deployment.evidence import (
        read_json_evidence,
        verify_artifact_identity,
        verify_control_dt,
    )
    from supermimic.deployment.trajectory import open_trajectory, validate_reference_clock

    result, _ = read_json_evidence(run / "result.json")
    if result["status"] != "complete" or result["auto_reset"]:
        raise ValueError(f"Incompatible deployment: {run}")
    with open_trajectory(run / "trajectory.npz", fields=("reference_frame",)) as recorded:
        identity = verify_artifact_identity(
            result, motion_sha256=motion_sha256, trajectory_sha256=recorded.sha256
        )
        if identity["status"] != "verified":
            raise ValueError(f"Missing deployment artifact fingerprints: {run}")
        frames = recorded.arrays["reference_frame"]
        if recorded.states.shape[1] < 36 or len(recorded.states) != len(frames) + 1:
            raise ValueError(f"Trajectory clock or position layout discontinuity: {run}")
        validate_reference_clock(recorded, motion_frames, result["reference_goal_offset"])
        verify_control_dt(recorded.control_dt, result=result)
    return result, recorded


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

        from supermimic.artifacts import write_json
        from supermimic.deployment.reference import heading_error

        motion_hash, position, orientation, joints = _comparison_reference(args.motion)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        axes[0, 0].plot(position[:, 0], position[:, 1], color="black", label="reference", alpha=0.6)
        report = {
            "motion_sha256": motion_hash,
            "source": "recorded actual physical states",
            "runs": [],
        }
        for label, run in zip(args.labels, args.runs, strict=True):
            result, recorded = _comparison_run(run, motion_hash, len(joints))
            states, frames = recorded.states, recorded.arrays["reference_frame"]
            dt, digest = recorded.control_dt, recorded.sha256
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
            del recorded, states, frames
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
