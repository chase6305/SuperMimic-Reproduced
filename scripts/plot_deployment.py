"""Inspect continuous G1 motion tracking using recorded physical states and forces."""

from __future__ import annotations

import argparse
from pathlib import Path


def _plot_inputs(bundle: Path, trajectory: Path, motion: Path) -> dict:
    """Prepare diagnostic arrays without requiring a successful or complete run."""
    import numpy as np

    from supermimic.deployment.evidence import (
        read_json_evidence,
        read_motion_evidence,
        read_optional_result,
        verify_artifact_identity,
        verify_control_dt,
    )
    from supermimic.deployment.trajectory import open_trajectory, validate_reference_clock

    contract, contract_sha = read_json_evidence(bundle / "contract.json")
    with open_trajectory(trajectory, fields=("reference_frame", "physical_contacts")) as recorded:
        states = recorded.states
        frames = recorded.arrays["reference_frame"]
        contacts = recorded.arrays["physical_contacts"]
    if not len(frames):
        raise ValueError("Trajectory has no executed controls")
    if len(states) != len(frames) + 1:
        raise ValueError("Trajectory states must contain the initial state and reached states")
    reference = read_motion_evidence(motion)
    names = reference.arrays["body_names"].tolist()
    root_body = names.index("pelvis")
    reference_position = reference.arrays["body_pos_w"][:, root_body].copy()
    reference_joints = reference.arrays["joint_pos"]
    result, result_sha = read_optional_result(trajectory)
    identity = verify_artifact_identity(
        result,
        trajectory_sha256=recorded.sha256,
        motion_sha256=reference.sha256,
        contract_sha256=contract_sha,
        checkpoint_sha256=contract["checkpoint_sha256"],
    )
    verify_control_dt(recorded.control_dt, contract=contract, result=result)
    if result is not None and "reference_goal_offset" in result:
        offset, clock_source = result["reference_goal_offset"], "result.json"
        if "first_goal_frame" in contract and offset != contract["first_goal_frame"]:
            raise ValueError("Reference goal offset differs from the contract")
    elif "first_goal_frame" in contract:
        offset, clock_source = contract["first_goal_frame"], "contract.json"
    else:
        offset, clock_source = int(frames[0]), "first saved reference frame"
    validate_reference_clock(recorded, len(reference_joints), offset)
    initial = contract.get("initial_reference_state_frame", 0)
    if type(initial) is not int or not 0 <= initial < len(reference_joints):
        raise ValueError("Initial reference state frame is outside the motion")
    root = contract["root_qpos_index"]
    if type(root) is not int or not 0 <= root <= states.shape[1] - 3:
        raise ValueError("Root position indices are outside the saved states")
    for side in ("left", "right"):
        joint = contract["joint_names"].index(side + "_knee_joint")
        actual_index = contract["joint_qpos_indices"][joint]
        if (
            type(actual_index) is not int
            or not 0 <= actual_index < states.shape[1]
            or joint >= reference_joints.shape[1]
        ):
            raise ValueError("Knee joint indices are outside the recorded/reference states")
    indices = np.empty(len(frames) + 1, dtype=np.intp)
    indices[0], indices[1:] = initial, frames
    return {
        "contract": contract,
        "recorded": recorded,
        "states": states,
        "frames": frames,
        "contacts": contacts,
        "dt": recorded.control_dt,
        "indices": indices,
        "reference_position": reference_position,
        "reference_joints": reference_joints,
        "evidence": {
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "contract_sha256": contract_sha,
            "trajectory_sha256": recorded.sha256,
            "motion_sha256": reference.sha256,
            "result_sha256": result_sha,
            "artifact_identity": identity,
            "reference_clock": {"offset": int(offset), "source": clock_source},
            "physical_contact_columns_verified": recorded.contact_layout_verified,
            "recorded_status": None if result is None else result.get("status"),
            "recorded_auto_reset": None if result is None else result.get("auto_reset"),
        },
    }


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

        from supermimic.artifacts import write_json

        inputs = _plot_inputs(args.bundle, args.trajectory, args.motion)
        contract, states, frames, contacts, dt, indices, reference_position, reference_joints = (
            inputs[key]
            for key in (
                "contract",
                "states",
                "frames",
                "contacts",
                "dt",
                "indices",
                "reference_position",
                "reference_joints",
            )
        )
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
        contact_layout_verified = inputs["recorded"].contact_layout_verified
        if contact_layout_verified and len(contacts):
            first_contacts = contacts[:, 0] <= 10
            for column, label in ((3, "left"), (6, "right")):
                axes[2, 0].plot(
                    contacts[first_contacts, 0],
                    contacts[first_contacts, column],
                    label=label,
                    linewidth=0.8,
                )
        axes[2, 0].set(
            title="Recorded foot-ground vertical force: first 10 seconds"
            if contact_layout_verified
            else "Contact layout unverified: force curves omitted",
            xlabel="time (s)",
            ylabel="N",
        )
        if contact_layout_verified and len(contacts):
            axes[2, 0].legend()
        error = np.linalg.norm(position[:, :2] - reference_position[indices, :2], axis=-1)
        axes[2, 1].plot(time, error)
        axes[2, 1].set(title="Root XY tracking error", xlabel="time (s)", ylabel="m")
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        figure.suptitle(
            f"G1 policy iteration {contract['checkpoint_iteration']}: saved trajectory; "
            f"artifact identity {inputs['evidence']['artifact_identity']['status']}"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)
        write_json(
            args.output.with_suffix(".json"),
            {
                **inputs["evidence"],
                "simulated_seconds": len(frames) * dt,
                "actual_xy_distance_m": float(
                    np.linalg.norm(np.diff(position[:, :2], axis=0), axis=-1).sum()
                ),
                "reference_xy_distance_full_clip_m": float(
                    np.linalg.norm(np.diff(reference_position[:, :2], axis=0), axis=-1).sum()
                ),
                "source": "saved trajectory states; no new simulation",
            },
        )
    print(args.output)


if __name__ == "__main__":
    main()
