"""Inspect continuous G1 motion tracking using recorded physical states and forces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from supermimic.cpu_resources import cpu_guard

    with cpu_guard(args.output):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from supermimic.artifacts import sha256, write_json

        contract = json.loads((args.bundle / "contract.json").read_text())
        with np.load(args.trajectory, allow_pickle=False) as recorded:
            states = recorded["states"]
            frames = recorded["reference_frame"]
            contacts = recorded["physical_contacts"]
            dt = float(recorded["control_dt"])
        with np.load(args.motion, allow_pickle=False) as reference:
            root = reference["body_names"].tolist().index("pelvis")
            reference_position = reference["body_pos_w"][:, root].copy()
            reference_joints = reference["joint_pos"]
        if not len(frames):
            raise ValueError("Trajectory has no executed controls")
        indices = np.concatenate([[0], frames])
        time = np.arange(len(states)) * dt
        root = contract["root_qpos_index"]
        position = states[:, root : root + 3]
        figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        axes[0, 0].plot(reference_position[:, 0], reference_position[:, 1], label="reference")
        axes[0, 0].plot(position[:, 0], position[:, 1], label="actual", alpha=0.8)
        axes[0, 0].set(title="Entire recorded XY path", xlabel="x (m)", ylabel="y (m)")
        axes[0, 0].set_aspect("equal", adjustable="datalim")
        axes[0, 0].legend()
        axes[0, 1].plot(time, reference_position[indices, 2], label="reference")
        axes[0, 1].plot(time, position[:, 2], label="actual", alpha=0.7)
        axes[0, 1].set(title="Root height", xlabel="time (s)", ylabel="m")
        first = time <= 10
        for axis, side in zip(axes[1], ("left", "right"), strict=True):
            joint = contract["joint_names"].index(side + "_knee_joint")
            actual_index = contract["joint_qpos_indices"][joint]
            axis.plot(time[first], reference_joints[indices[first], joint], label="reference")
            axis.plot(time[first], states[first, actual_index], label="actual", alpha=0.8)
            axis.set(
                title=f"{side.title()} knee: first 10 seconds", xlabel="time (s)", ylabel="rad"
            )
        if len(contacts):
            first_contacts = contacts[:, 0] <= 10
            for column, label in ((3, "left"), (6, "right")):
                axes[2, 0].plot(
                    contacts[first_contacts, 0],
                    contacts[first_contacts, column],
                    label=label,
                    linewidth=0.8,
                )
        axes[2, 0].set(
            title="Actual foot-ground vertical force: first 10 seconds",
            xlabel="time (s)",
            ylabel="N",
        )
        axes[2, 0].legend()
        error = np.linalg.norm(position[:, :2] - reference_position[indices, :2], axis=-1)
        axes[2, 1].plot(time, error)
        axes[2, 1].set(title="Root XY tracking error", xlabel="time (s)", ylabel="m")
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        figure.suptitle(
            f"G1 policy iteration {contract['checkpoint_iteration']}: actual MuJoCo trajectory, no reset"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)
        write_json(
            args.output.with_suffix(".json"),
            {
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "trajectory_sha256": sha256(args.trajectory),
                "motion_sha256": sha256(args.motion),
                "simulated_seconds": len(frames) * dt,
                "actual_xy_distance_m": float(
                    np.linalg.norm(np.diff(position[:, :2], axis=0), axis=-1).sum()
                ),
                "reference_xy_distance_full_clip_m": float(
                    np.linalg.norm(np.diff(reference_position[:, :2], axis=0), axis=-1).sum()
                ),
                "source": "saved physical trajectory; no new simulation",
            },
        )
    print(args.output)


if __name__ == "__main__":
    main()
