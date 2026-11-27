"""Single-robot CPU MuJoCo deployment with uninterrupted trajectory recording."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from supermimic.artifacts import sha256, snapshot_file, write_json
from supermimic.data.schema import validate_motion
from supermimic.deployment.core import MotionReference, PolicyRunner, gravity, rotation
from supermimic.deployment.trajectory import PHYSICAL_CONTACT_COLUMNS
from supermimic.deployment.validation import validate_model_contract
from supermimic.interfaces import ACT_DIM
from supermimic.runtime import source_manifest


def _record_contact_row(model, data, floor_geoms, foot_sides, local_force, row):
    """Record current solver forces in contact order before any further stepping."""
    row[0] = data.time
    forces = row[1:10].reshape(3, 3)
    forces.fill(0)
    nonfoot_count = 0
    max_depth = 0.0
    contacts = data.contact
    frames, distances = contacts.frame, contacts.dist
    for contact_id, (one, two) in enumerate(contacts.geom):
        one, two = int(one), int(two)
        one_is_floor, two_is_floor = one in floor_geoms, two in floor_geoms
        if not (one_is_floor or two_is_floor):
            continue
        mujoco.mj_contactForce(model, data, contact_id, local_force)
        world_force = frames[contact_id].reshape(3, 3).T @ local_force[:3]
        other = two if one_is_floor else one
        if two_is_floor:
            world_force = -world_force
        side = foot_sides.get(other, 2)
        forces[side] += world_force
        if side == 2:
            nonfoot_count += int(np.linalg.norm(world_force) > 1e-3)
            max_depth = max(max_depth, float(-distances[contact_id]))
    row[10:] = nonfoot_count, max_depth, data.ncon


def deploy(
    bundle: Path, motion: Path, output: Path, steps: int | None = None, viewer: bool = False
) -> dict:
    runner = PolicyRunner(bundle)
    c = runner.contract
    if c["packages"]["mujoco"] != mujoco.__version__:
        raise ValueError("MJB deployment requires the same MuJoCo version as export")
    model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
    validate_model_contract(model, c)
    motion = Path(motion)
    validate_motion(motion)
    existing = output.expanduser().resolve().parent
    while not existing.exists():
        existing = existing.parent
    with TemporaryDirectory(prefix=".supermimic-deploy-", dir=existing) as temporary:
        pinned_motion = Path(temporary) / motion.name
        snapshot_file(motion, pinned_motion)
        return _deploy(runner, model, bundle, pinned_motion, output, steps, viewer)


def _deploy(
    runner: PolicyRunner,
    model: mujoco.MjModel,
    bundle: Path,
    motion: Path,
    output: Path,
    steps: int | None,
    viewer: bool,
) -> dict:
    c = runner.contract
    reference = MotionReference(motion, runner.default_q)
    data = mujoco.MjData(model)
    qadr = np.asarray(c["joint_qpos_indices"], dtype=np.intp)
    vadr = np.asarray(c["joint_qvel_indices"], dtype=np.intp)
    root_q, root_v = c["root_qpos_index"], c["root_qvel_index"]
    root_rotation = slice(root_q + 3, root_q + 7)
    body_omega = slice(root_v + 3, root_v + 6)
    data.qpos[root_q : root_q + 3] = reference.root_pos[0]
    data.qpos[root_rotation] = reference.root_quat[0]
    limits = np.asarray(c["joint_limits"])
    center = limits.mean(axis=1)
    half = (limits[:, 1] - limits[:, 0]) * c.get("initial_joint_soft_limit_factor", 0.9) / 2
    data.qpos[qadr] = np.clip(reference.q[0], center - half, center + half)
    data.qvel[root_v : root_v + 3] = reference.root_lin[0]
    data.qvel[body_omega] = (
        rotation(reference.root_quat[0]).inv().apply(reference.root_ang[0])
    )
    data.qvel[vadr] = reference.qd[0]
    mujoco.mj_forward(model, data)
    projected_gravity = gravity(data.qpos[root_rotation])
    anchor_id = model.body("robot/" + c["anchor_body"]).id
    anchor_targets = rotation(reference.anchor_quat) if c["anchor_obs"] else None
    actuator_order = np.asarray(c["actuator_joint_indices"], dtype=np.intp)
    goal_offset = c.get("first_goal_frame", 1)
    future_frames = c["future_frames"]
    goal = np.empty(c["goal_dim"], dtype=np.float32)
    motion_goal_dim = future_frames * reference.features.shape[1]
    goal_rows = goal[:motion_goal_dim].reshape(future_frames, -1)
    decimation = c["decimation"]
    requested = steps if steps is not None else reference.frames - 1
    if requested < 1 or requested > 10_000_000:
        raise ValueError("Deployment step budget must be within [1, 10000000]")
    state_width = model.nq + model.nv
    contact_width = len(PHYSICAL_CONTACT_COLUMNS)
    float64_bytes = np.dtype(np.float64).itemsize
    per_step_bytes = (
        (state_width + decimation * contact_width) * float64_bytes
        + ACT_DIM * np.dtype(np.float32).itemsize
        + np.dtype(np.int64).itemsize
    )
    initial_state_bytes = state_width * float64_bytes
    if initial_state_bytes + requested * per_step_bytes > 512 * 1024**2:
        raise ValueError("Trajectory exceeds the 512 MiB recording budget")
    output.mkdir(parents=True, exist_ok=False)
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
        "reference_goal_offset": goal_offset,
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("mujoco.py", "core.py", "metrics.py")
        },
        # Whole-package Python source inventory, not an exact imported-module trace.
        "implementation_package_sha256": source_manifest(),
    }
    write_json(output / "result.json", record)
    states = np.empty((requested + 1, state_width), dtype=np.float64)
    actions = np.empty((requested, ACT_DIM), dtype=np.float32)
    frames = np.empty(requested, dtype=np.int64)
    contacts = np.empty((requested * decimation, contact_width), dtype=np.float64)
    contact_steps = 0
    floors = {i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
    side_by_body = {
        model.body("robot/" + side + "_ankle_roll_link").id: index
        for index, side in enumerate(("left", "right"))
    }
    foot_sides = {
        geom: side_by_body[body]
        for geom, body in enumerate(model.geom_bodyid)
        if body in side_by_body
    }
    local_force = np.empty(6)
    warnings = data.warning.number
    bad_qpos = mujoco.mjtWarning.mjWARN_BADQPOS
    bad_qvel = mujoco.mjtWarning.mjWARN_BADQVEL
    bad_qacc = mujoco.mjtWarning.mjWARN_BADQACC
    states[0, : model.nq] = data.qpos
    states[0, model.nq :] = data.qvel
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
                frame = min(step + goal_offset, reference.frames - 1)
                available = min(future_frames, reference.frames - frame)
                goal_rows[:available] = reference.features[frame : frame + available]
                if available < future_frames:
                    goal_rows[available:] = goal_rows[available - 1]
                if anchor_targets is not None:
                    world_to_anchor = rotation(data.xquat[anchor_id]).inv()
                    goal[motion_goal_dim : motion_goal_dim + 3] = world_to_anchor.apply(
                        reference.anchor_pos[frame] - data.xpos[anchor_id]
                    )
                    relative = world_to_anchor * anchor_targets[frame]
                    goal[motion_goal_dim + 3 :] = relative.as_matrix()[:, :2].reshape(-1)
                if window is not None:
                    # A passive viewer can modify the state between controls.
                    projected_gravity = gravity(data.qpos[root_rotation])
                action, target = runner._act_with_gravity(
                    data.qpos[qadr], data.qvel[vadr], projected_gravity, data.qvel[body_omega], goal
                )
                data.ctrl[:] = target[actuator_order]
                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                    if warnings[bad_qpos] or warnings[bad_qvel] or warnings[bad_qacc]:
                        raise RuntimeError(
                            "MuJoCo reported invalid state/acceleration; rollout rejected"
                        )
                    _record_contact_row(
                        model, data, floors, foot_sides, local_force, contacts[contact_steps]
                    )
                    contact_steps += 1
                mujoco.mj_forward(model, data)
                states[step + 1, : model.nq] = data.qpos
                states[step + 1, model.nq :] = data.qvel
                actions[step] = action
                frames[step] = frame
                actual_steps = step + 1
                if not np.isfinite(states[actual_steps]).all():
                    failure = "non_finite_physics"
                    break
                height_error = abs(reference.root_pos[frame, 2] - data.qpos[root_q + 2])
                projected_gravity = gravity(data.qpos[root_rotation])
                tilt_error = abs(reference.root_gravity_z[frame] - projected_gravity[2])
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
            physical_contact_columns=np.array(PHYSICAL_CONTACT_COLUMNS),
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
