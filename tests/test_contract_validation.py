"""Reject metadata drift before policy deserialization or physical control."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from supermimic.artifacts import write_json
from supermimic.deployment.contract import build_model_and_contract
from supermimic.deployment.core import PolicyRunner
from supermimic.deployment.validation import validate_contract, validate_model_contract


@pytest.fixture(scope="module")
def model_and_contract():
    model, value = build_model_and_contract(history=3, future_frames=4, anchor_obs=True)
    assert value["control_dt"] == model.opt.timestep * value["decimation"]
    return model, value


@pytest.fixture(scope="module")
def contract(model_and_contract):
    _, value = model_and_contract
    return value


def test_generated_contract_supports_nondefault_input_sizes(contract):
    validate_contract(contract)
    assert contract["observation_dim"] == 279
    assert contract["goal_dim"] == 281


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quaternion_order", "xyzw"),
        ("root_angular_velocity_frame", "world"),
        ("history_order", "newest_to_oldest"),
        ("history_layout", "frame_major"),
        ("history_reset", "zeros"),
        ("observation_terms", [["joint_pos", 93]]),
        ("observation_dim", 930),
        ("goal_dim", 680),
        ("anchor_obs", 1),
        ("history", True),
        ("future_frames", 0),
        ("decimation", 0),
        ("control_dt", 0.01),
        ("physics_dt", float("nan")),
        ("initial_reference_state_frame", 1),
        ("initial_reference_state_frame", False),
        ("first_goal_frame", 0),
        ("first_goal_frame", 2),
        ("first_goal_frame", -1),
        ("initial_joint_soft_limit_factor", float("nan")),
        ("initial_joint_soft_limit_factor", float("inf")),
        ("initial_joint_soft_limit_factor", 0),
        ("initial_joint_soft_limit_factor", -0.1),
        ("initial_joint_soft_limit_factor", 1.1),
        ("initial_joint_soft_limit_factor", True),
        ("action_scale", [float("inf")] * 29),
        ("default_joint_pos", [0.0] * 28),
        ("actuator_joint_indices", [0] * 29),
        ("joint_qpos_indices", list(range(28)) + [-1]),
        ("joint_limits", [[1.0, -1.0]] * 29),
    ],
)
def test_metadata_drift_is_rejected(contract, field, value):
    changed = deepcopy(contract)
    changed[field] = value
    with pytest.raises(ValueError, match=field):
        validate_contract(changed)


def test_contract_requires_reference_clock_even_with_consistent_substeps(contract):
    changed = deepcopy(contract)
    changed["control_dt"] *= 2
    changed["physics_dt"] *= 2
    with pytest.raises(ValueError, match="control_dt"):
        validate_contract(changed)


def test_early_schema_offsets_use_existing_defaults_and_allow_provenance(contract):
    early = deepcopy(contract)
    del early["initial_reference_state_frame"]
    del early["first_goal_frame"]
    del early["initial_joint_soft_limit_factor"]
    early["additional_provenance"] = {"source": "older schema-v1 export"}
    validate_contract(early)


@pytest.mark.parametrize("factor", [0.5, 1.0])
def test_valid_custom_initial_soft_limit_factor_is_accepted(contract, factor):
    changed = deepcopy(contract)
    changed["initial_joint_soft_limit_factor"] = factor
    validate_contract(changed)


def test_compiled_model_matches_generated_contract(model_and_contract):
    model, contract = model_and_contract
    validate_model_contract(model, contract)


def test_consistent_contract_timestep_must_match_actual_model(model_and_contract):
    model, contract = model_and_contract
    changed = deepcopy(contract)
    changed["physics_dt"] *= 2
    changed["decimation"] //= 2
    validate_contract(changed)
    with pytest.raises(ValueError, match="physics_dt"):
        validate_model_contract(model, changed)


@pytest.mark.parametrize(
    "field",
    [
        "joint_qpos_indices",
        "joint_qvel_indices",
        "actuator_joint_indices",
        "root_qpos_index",
        "root_qvel_index",
    ],
)
def test_contract_addresses_must_match_compiled_model(model_and_contract, field):
    model, contract = model_and_contract
    changed = deepcopy(contract)
    if isinstance(changed[field], list):
        changed[field][0], changed[field][1] = changed[field][1], changed[field][0]
    else:
        changed[field] += 1
    validate_contract(changed)
    with pytest.raises(ValueError, match=field):
        validate_model_contract(model, changed)


@pytest.mark.parametrize("dimension", ["nq", "nv", "nu"])
def test_model_dimensions_must_match_g1_layout(model_and_contract, dimension):
    model, contract = model_and_contract
    dimensions = {name: getattr(model, name) for name in ("nq", "nv", "nu")}
    dimensions[dimension] += 1
    incompatible = SimpleNamespace(**dimensions)
    with pytest.raises(ValueError, match=dimension):
        validate_model_contract(incompatible, contract)


def test_deployment_checks_model_addresses_before_allocating_state(
    tmp_path, model_and_contract, monkeypatch
):
    from supermimic.deployment import mujoco as runtime

    model, contract = model_and_contract
    changed = deepcopy(contract)
    changed["packages"] = {"mujoco": runtime.mujoco.__version__}
    changed["joint_qpos_indices"][0], changed["joint_qpos_indices"][1] = (
        changed["joint_qpos_indices"][1],
        changed["joint_qpos_indices"][0],
    )
    runtime.mujoco.mj_saveModel(model, str(tmp_path / "model.mjb"))
    runner = SimpleNamespace(contract=changed, default_q=[0.0] * 29)
    monkeypatch.setattr(runtime, "PolicyRunner", lambda bundle: runner)
    monkeypatch.setattr(runtime, "MotionReference", lambda motion, default_q: None)

    def unexpected_state(*args, **kwargs):
        raise AssertionError("Physical state must not be allocated with incorrect addresses")

    monkeypatch.setattr(runtime.mujoco, "MjData", unexpected_state)
    output = tmp_path / "rollout"
    with pytest.raises(ValueError, match="joint_qpos_indices"):
        runtime.deploy(tmp_path, tmp_path / "unused.npz", output)
    assert not output.exists()


def test_runner_checks_contract_before_loading_artifacts(tmp_path, contract, monkeypatch):
    changed = deepcopy(contract)
    changed["history_layout"] = "frame_major"
    write_json(tmp_path / "contract.json", changed)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Policy must not be loaded before validating its interface")

    monkeypatch.setattr("torch.jit.load", unexpected_load)
    with pytest.raises(ValueError, match="history_layout"):
        PolicyRunner(tmp_path)
