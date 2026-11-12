"""Derive the deployment contract from the same G1 scene used during training."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from supermimic.deployment.validation import validate_contract
from supermimic.interfaces import ACT_DIM, G1_JOINT_NAMES, OBS_TERMS, goal_dim, observation_dim


def build_model_and_contract(
    history: int = 10, future_frames: int = 10, anchor_obs: bool = False
) -> tuple[mujoco.MjModel, dict]:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE
    from mjlab.scene import Scene

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg

    cfg = make_env_cfg(
        SuperMimicEnvCfgArgs(
            npz_files=[],
            num_envs=1,
            history_length=history,
            future_frames=future_frames,
            anchor_obs=anchor_obs,
        )
    )
    scene = Scene(cfg.scene, "cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    # Match the deterministic evaluator's startup event.
    for i in range(model.ngeom):
        name = model.geom(i).name.rsplit("/", 1)[-1]
        if re.fullmatch(r"(left|right)_foot[1-7]_collision", name):
            model.geom_friction[i, 0] = 1.0
    joint_ids = [model.joint("robot/" + name).id for name in G1_JOINT_NAMES]
    if len(set(joint_ids)) != 29 or model.nu != 29:
        raise ValueError("The compiled model does not have 29 unique actuated G1 joints")
    qadr = model.jnt_qposadr[joint_ids]
    vadr = model.jnt_dofadr[joint_ids]
    default_q = model.key_qpos[0, qadr]
    actuator_map = [joint_ids.index(int(j)) for j in model.actuator_trnid[:, 0]]
    scales = []
    kp = np.zeros(29)
    kd = np.zeros(29)
    effort = np.zeros(29)
    for name in G1_JOINT_NAMES:
        matches = [
            value for pattern, value in G1_ACTION_SCALE.items() if re.fullmatch(pattern, name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous action scale for {name}")
        scales.append(matches[0])
    for actuator, joint_index in enumerate(actuator_map):
        kp[joint_index] = model.actuator_gainprm[actuator, 0]
        kd[joint_index] = -model.actuator_biasprm[actuator, 2]
        effort[joint_index] = model.actuator_forcerange[actuator, 1]
    contract = {
        "schema_version": 1,
        "robot": "unitree-g1-29dof",
        "algorithm": "cuboi-supermimic",
        "joint_names": list(G1_JOINT_NAMES),
        "quaternion_order": "wxyz",
        "root_linear_velocity_frame": "world",
        "root_angular_velocity_frame": "body",
        "control_dt": float(model.opt.timestep) * cfg.decimation,
        "physics_dt": float(model.opt.timestep),
        "decimation": cfg.decimation,
        "observation_terms": [list(term) for term in OBS_TERMS],
        "history": history,
        "history_order": "oldest_to_newest",
        "history_layout": "term_major",
        "history_reset": "repeat_first_observation",
        "observation_dim": observation_dim(history),
        "future_frames": future_frames,
        "goal_dim": goal_dim(future_frames, anchor_obs),
        "anchor_obs": anchor_obs,
        "anchor_body": "torso_link",
        "action_dim": ACT_DIM,
        "default_joint_pos": default_q.tolist(),
        "action_scale": scales,
        "action_semantics": "joint_target = default_joint_pos + action_scale * raw_action",
        "action_clipping": None,
        "initial_reference_state_frame": 0,
        "first_goal_frame": 1,
        "initial_joint_soft_limit_factor": 0.9,
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "effort_limit": effort.tolist(),
        "joint_qpos_indices": qadr.tolist(),
        "joint_qvel_indices": vadr.tolist(),
        "joint_limits": model.jnt_range[joint_ids].tolist(),
        "actuator_joint_indices": actuator_map,
        "root_qpos_index": int(model.jnt_qposadr[model.joint("robot/floating_base_joint").id]),
        "root_qvel_index": int(model.jnt_dofadr[model.joint("robot/floating_base_joint").id]),
        "deployment_target": "mujoco_cpu",
        "hardware_validated": False,
    }
    validate_contract(contract)
    return model, contract
