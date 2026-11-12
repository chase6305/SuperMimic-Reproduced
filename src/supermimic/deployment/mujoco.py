"""Single-robot CPU MuJoCo deployment with uninterrupted trajectory recording."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from supermimic.artifacts import sha256, write_json
from supermimic.deployment.core import MotionReference, PolicyRunner, gravity, rotation
from supermimic.deployment.validation import validate_model_contract


def deploy(
    bundle: Path, motion: Path, output: Path, steps: int | None = None, viewer: bool = False
) -> dict:
    runner = PolicyRunner(bundle)
    c = runner.contract
    if c["packages"]["mujoco"] != mujoco.__version__:
        raise ValueError("MJB deployment requires the same MuJoCo version as export")
    reference = MotionReference(motion, runner.default_q)
    model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
    validate_model_contract(model, c)
    data = mujoco.MjData(model)
    qadr, vadr = c["joint_qpos_indices"], c["joint_qvel_indices"]
    root_q, root_v = c["root_qpos_index"], c["root_qvel_index"]
    data.qpos[root_q : root_q + 3] = reference.root_pos[0]
    data.qpos[root_q + 3 : root_q + 7] = reference.root_quat[0]
    limits = np.asarray(c["joint_limits"])
    center = limits.mean(axis=1)
    half = (limits[:, 1] - limits[:, 0]) * c.get("initial_joint_soft_limit_factor", 0.9) / 2
    data.qpos[qadr] = np.clip(reference.q[0], center - half, center + half)
    data.qvel[root_v : root_v + 3] = reference.root_lin[0]
    data.qvel[root_v + 3 : root_v + 6] = (
        rotation(reference.root_quat[0]).inv().apply(reference.root_ang[0])
    )
    data.qvel[vadr] = reference.qd[0]
    mujoco.mj_forward(model, data)
    anchor_id = model.body("robot/" + c["anchor_body"]).id
    output.mkdir(parents=True, exist_ok=False)
    requested = steps if steps is not None else reference.frames - 1
    if requested < 1 or requested > 10_000_000:
        raise ValueError("Deployment step budget must be within [1, 10000000]")
    if requested * (model.nq + model.nv + 29 + 4 * 13) * 8 > 512 * 1024**2:
        raise ValueError("Trajectory exceeds the 512 MiB recording budget")
    record = {
        "status": "running",
        "backend": "mujoco_cpu",
        "hardware_deployed": False,
        "checkpoint_sha256": c["checkpoint_sha256"],
        "motion_sha256": sha256(motion),
        "contract_sha256": sha256(bundle / "contract.json"),
        "requested_steps": requested,
        "control_dt": c["control_dt"],
        "requested_duration_s": requested * c["control_dt"],
        "auto_reset": False,
        "reference_goal_offset": c.get("first_goal_frame", 1),
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("mujoco.py", "core.py", "metrics.py")
        },
    }
    write_json(output / "result.json", record)
    states = np.empty((requested + 1, model.nq + model.nv), dtype=np.float64)
    actions = np.empty((requested, 29), dtype=np.float32)
    frames = np.empty(requested, dtype=np.int64)
    contacts = np.empty((requested * c["decimation"], 13), dtype=np.float64)
    contact_steps = 0
    floors = {i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
    feet = [
        {
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body("robot/" + side + "_ankle_roll_link").id
        }
        for side in ("left", "right")
    ]
    states[0] = np.concatenate([data.qpos, data.qvel])
    actual_steps = 0
    failure = None
    max_height_error = max_tilt_error = max_joint_error = 0.0
    started = time.monotonic()
    last_progress = started
    if viewer:
        from mujoco import viewer as mjviewer

        viewer_context = mjviewer.launch_passive(model, data)
    else:
        viewer_context = nullcontext(None)
    try:
        with viewer_context as window:
            for step in range(requested):
                frame = min(step + c.get("first_goal_frame", 1), reference.frames - 1)
                goal = reference.goal(frame, c["future_frames"])
                if c["anchor_obs"]:
                    robot_anchor = rotation(data.xquat[anchor_id])
                    delta = robot_anchor.inv().apply(
                        reference.anchor_pos[frame] - data.xpos[anchor_id]
                    )
                    relative = robot_anchor.inv() * rotation(reference.anchor_quat[frame])
                    goal = np.concatenate([goal, delta, relative.as_matrix()[:, :2].reshape(-1)])
                action, target = runner.act(
                    data.qpos[qadr],
                    data.qvel[vadr],
                    data.qpos[root_q + 3 : root_q + 7],
                    data.qvel[root_v + 3 : root_v + 6],
                    goal,
                )
                data.ctrl[:] = target[c["actuator_joint_indices"]]
                for _ in range(c["decimation"]):
                    mujoco.mj_step(model, data)
                    for bad in (
                        mujoco.mjtWarning.mjWARN_BADQPOS,
                        mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC,
                    ):
                        if data.warning[bad].number:
                            raise RuntimeError(
                                "MuJoCo reported invalid state/acceleration; rollout rejected"
                            )
                    forces = np.zeros((3, 3))  # left foot, right foot, non-foot ground
                    nonfoot_count = 0
                    max_depth = 0.0
                    local_force = np.zeros(6)
                    for contact_id in range(data.ncon):
                        contact = data.contact[contact_id]
                        one, two = map(int, contact.geom)
                        if one not in floors and two not in floors:
                            continue
                        mujoco.mj_contactForce(model, data, contact_id, local_force)
                        world_force = contact.frame.reshape(3, 3).T @ local_force[:3]
                        other = two if one in floors else one
                        if two in floors:
                            world_force = -world_force
                        side = next((i for i, ids in enumerate(feet) if other in ids), 2)
                        forces[side] += world_force
                        if side == 2:
                            nonfoot_count += int(np.linalg.norm(world_force) > 1e-3)
                            max_depth = max(max_depth, float(-contact.dist))
                    contacts[contact_steps] = np.concatenate(
                        [[data.time], forces.reshape(-1), [nonfoot_count, max_depth, data.ncon]]
                    )
                    contact_steps += 1
                mujoco.mj_forward(model, data)
                states[step + 1] = np.concatenate([data.qpos, data.qvel])
                actions[step] = action
                frames[step] = frame
                actual_steps = step + 1
                if not np.isfinite(states[actual_steps]).all():
                    failure = "non_finite_physics"
                    break
                height_error = abs(reference.root_pos[frame, 2] - data.qpos[root_q + 2])
                tilt_error = abs(
                    gravity(reference.root_quat[frame])[2]
                    - gravity(data.qpos[root_q + 3 : root_q + 7])[2]
                )
                max_height_error = max(max_height_error, float(height_error))
                max_tilt_error = max(max_tilt_error, float(tilt_error))
                max_joint_error = max(
                    max_joint_error, float(np.abs(data.qpos[qadr] - reference.q[frame]).max())
                )
                now = time.monotonic()
                if now - last_progress >= 1:
                    write_json(
                        output / "progress.json",
                        {
                            "status": "running",
                            "steps": actual_steps,
                            "requested_steps": requested,
                            "simulated_seconds": actual_steps * c["control_dt"],
                            "elapsed_s": now - started,
                            "max_root_height_error_m": max_height_error,
                        },
                    )
                    last_progress = now
                if height_error > 0.25 or tilt_error > 0.8:
                    failure = "root_height" if height_error > 0.25 else "root_tilt"
                    break
                if window:
                    if not window.is_running():
                        failure = "viewer_closed"
                        break
                    window.sync()
                    wait = (step + 1) * c["control_dt"] - (time.monotonic() - started)
                    if wait > 0:
                        time.sleep(wait)
        record.update(
            status="complete",
            steps=actual_steps,
            simulated_seconds=actual_steps * c["control_dt"],
            failure=failure,
            completed_motion=actual_steps >= reference.frames - 1 and failure is None,
            max_root_height_error_m=max_height_error,
            max_root_tilt_error=max_tilt_error,
            max_joint_error_rad=max_joint_error,
        )
        from supermimic.deployment.metrics import airborne_metrics, tracking_metrics

        record["tracking"] = tracking_metrics(
            states[: actual_steps + 1], actions[:actual_steps], frames[:actual_steps], motion, c
        )
        physical = contacts[:contact_steps]
        record["contacts"] = {
            **airborne_metrics(physical, model.opt.timestep),
            "physics_substeps": contact_steps,
            "nonfoot_ground_force_substeps": int(np.count_nonzero(physical[:, 10] > 0)),
            "max_nonfoot_penetration_m": float(physical[:, 11].max()) if contact_steps else 0.0,
            "max_ground_force_N": float(
                np.linalg.norm(physical[:, 1:10].reshape(-1, 3, 3).sum(axis=1), axis=-1).max()
            )
            if contact_steps
            else 0.0,
            "source": "actual CPU MuJoCo solved forces immediately after each physics substep",
        }
        return record
    except BaseException as error:  # noqa: BLE001 - persist partial physical trajectories on failure
        record.update(status="failed", error=f"{type(error).__name__}: {error}", steps=actual_steps)
        raise
    finally:
        np.savez(
            output / "trajectory.npz",
            states=states[: actual_steps + 1],
            actions=actions[:actual_steps],
            reference_frame=frames[:actual_steps],
            control_dt=np.array(c["control_dt"]),
            joint_names=np.array(c["joint_names"]),
            physical_contacts=contacts[:contact_steps],
            physical_contact_columns=np.array(
                [
                    "time_s",
                    "left_fx",
                    "left_fy",
                    "left_fz",
                    "right_fx",
                    "right_fy",
                    "right_fz",
                    "nonfoot_fx",
                    "nonfoot_fy",
                    "nonfoot_fz",
                    "nonfoot_force_contacts",
                    "nonfoot_depth_m",
                    "all_contacts",
                ]
            ),
        )
        record.update(
            elapsed_s=time.monotonic() - started,
            trajectory_sha256=sha256(output / "trajectory.npz"),
        )
        write_json(output / "result.json", record)
        write_json(
            output / "progress.json",
            {
                "status": record["status"],
                "steps": actual_steps,
                "requested_steps": requested,
                "simulated_seconds": actual_steps * c["control_dt"],
                "failure": record.get("failure"),
                "elapsed_s": record["elapsed_s"],
            },
        )
