"""Validate the supported deployment interface before loading a policy or model."""

from __future__ import annotations

import math
from collections.abc import Mapping

from supermimic.interfaces import (
    ACT_DIM,
    CONTROL_DT,
    G1_JOINT_NAMES,
    OBS_TERMS,
    goal_dim,
    observation_dim,
)


def validate_contract(contract: Mapping) -> None:
    """Accept schema-v1 G1 contracts with a consistent policy and control layout.

    Extra provenance fields are allowed. File hashes are checked by the bundle
    reader; this checks the metadata that determines how those files are used.
    """
    fixed = {
        "schema_version": 1,
        "robot": "unitree-g1-29dof",
        "joint_names": list(G1_JOINT_NAMES),
        "quaternion_order": "wxyz",
        "root_linear_velocity_frame": "world",
        "root_angular_velocity_frame": "body",
        "observation_terms": [list(term) for term in OBS_TERMS],
        "history_order": "oldest_to_newest",
        "history_layout": "term_major",
        "history_reset": "repeat_first_observation",
        "anchor_body": "torso_link",
        "action_dim": ACT_DIM,
        "action_semantics": "joint_target = default_joint_pos + action_scale * raw_action",
    }
    for name, expected in fixed.items():
        if contract.get(name) != expected:
            raise ValueError(f"Unsupported policy contract {name}")
    # Early schema-v1 bundles omitted these fields; deployment has always used
    # frame zero for initialization and frame one for the first control target.
    for name, expected in (("initial_reference_state_frame", 0), ("first_goal_frame", 1)):
        value = contract.get(name, expected)
        if type(value) is not int or value != expected:
            raise ValueError(f"Unsupported policy contract {name}")
    for name in ("history", "future_frames", "decimation"):
        value = contract.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid policy contract {name}: expected a positive integer")
    if type(contract.get("anchor_obs")) is not bool:
        raise ValueError("Invalid policy contract anchor_obs: expected a boolean")
    dimensions = {
        "observation_dim": observation_dim(contract["history"]),
        "goal_dim": goal_dim(contract["future_frames"], contract["anchor_obs"]),
    }
    for name, expected in dimensions.items():
        if type(contract.get(name)) is not int or contract[name] != expected:
            raise ValueError(f"Inconsistent policy contract {name}")

    def finite(value) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    factor = contract.get("initial_joint_soft_limit_factor", 0.9)
    if not finite(factor) or not 0 < factor <= 1:
        raise ValueError("Invalid policy contract initial_joint_soft_limit_factor")

    for name in ("control_dt", "physics_dt"):
        if not finite(contract.get(name)) or contract[name] <= 0:
            raise ValueError(f"Invalid policy contract {name}")
    if not math.isclose(contract["control_dt"], CONTROL_DT, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Unsupported policy contract control_dt: expected 50 Hz controls")
    if not math.isclose(
        contract["control_dt"],
        contract["physics_dt"] * contract["decimation"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("Inconsistent policy contract control_dt/physics_dt/decimation")
    for name in ("default_joint_pos", "action_scale", "kp", "kd", "effort_limit"):
        values = contract.get(name)
        if (
            not isinstance(values, list)
            or len(values) != ACT_DIM
            or any(not finite(value) for value in values)
            or (name != "default_joint_pos" and any(value <= 0 for value in values))
        ):
            raise ValueError(f"Invalid policy contract {name}")
    for name in ("joint_qpos_indices", "joint_qvel_indices", "actuator_joint_indices"):
        values = contract.get(name)
        if (
            not isinstance(values, list)
            or len(values) != ACT_DIM
            or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != ACT_DIM
        ):
            raise ValueError(f"Invalid policy contract {name}")
        if name == "actuator_joint_indices" and sorted(values) != list(range(ACT_DIM)):
            raise ValueError("Invalid policy contract actuator_joint_indices")
    for name in ("root_qpos_index", "root_qvel_index"):
        value = contract.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid policy contract {name}")
    limits = contract.get("joint_limits")
    if (
        not isinstance(limits, list)
        or len(limits) != ACT_DIM
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(finite(value) for value in pair)
            or pair[0] >= pair[1]
            for pair in limits
        )
    ):
        raise ValueError("Invalid policy contract joint_limits")


def validate_model_contract(model, contract: Mapping) -> None:
    """Check compiled state/control addresses before allocating physical state.

    The model argument is deliberately duck typed: importing this metadata layer
    does not import MuJoCo. Policy identity and PD tuning are handled by their
    existing export and artifact checks; this verifies the model's address map.
    """
    validate_contract(contract)
    for name, expected in (("nq", 7 + ACT_DIM), ("nv", 6 + ACT_DIM), ("nu", ACT_DIM)):
        if getattr(model, name) != expected:
            raise ValueError(f"Model {name} differs from the G1 control contract")
    if not math.isclose(
        float(model.opt.timestep), contract["physics_dt"], rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("Model physics_dt differs from the control contract")
    try:
        joints = [model.joint("robot/" + name).id for name in G1_JOINT_NAMES]
        root = model.joint("robot/floating_base_joint").id
    except (KeyError, ValueError) as error:
        raise ValueError("Model joint names differ from the G1 control contract") from error
    for field, addresses in (
        ("joint_qpos_indices", model.jnt_qposadr),
        ("joint_qvel_indices", model.jnt_dofadr),
    ):
        if [int(addresses[joint]) for joint in joints] != contract[field]:
            raise ValueError(f"Model {field} differ from the control contract")
    for field, actual in (
        ("root_qpos_index", model.jnt_qposadr[root]),
        ("root_qvel_index", model.jnt_dofadr[root]),
    ):
        if int(actual) != contract[field]:
            raise ValueError(f"Model {field} differs from the control contract")
    actual_actuators = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    expected_actuators = [joints[index] for index in contract["actuator_joint_indices"]]
    if actual_actuators != expected_actuators:
        raise ValueError("Model actuator_joint_indices differ from the control contract")
