"""Independently replay saved clean commands from a full native integration state."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from supermimic.artifacts import sha256, snapshot_file
from supermimic.deployment.assessment import open_assessment_inputs
from supermimic.deployment.core import MotionReference
from supermimic.deployment.initialization import InitialPerturbation
from supermimic.deployment.mujoco import ground_contact_row, initialize_data
from supermimic.deployment.validation import validate_model_contract
from supermimic.numpy_io import inspect_npz_headers


def _read_arrays(path, shapes):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(shapes) or len(archive.files) != len(shapes):
            raise ValueError("Replay evidence contains unexpected arrays")
        headers = inspect_npz_headers(archive, tuple(shapes))
        for name, (shape, kind) in shapes.items():
            header = headers[name]
            if header.shape != shape or header.dtype != np.dtype(kind):
                raise ValueError(f"Replay evidence has invalid {name} shape or dtype")
        arrays = {name: archive[name] for name in shapes}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Replay evidence must be finite")
        return arrays


def _bitexact(actual, expected):
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and (actual.tobytes() == expected.tobytes())
    )


def verify_action_replay(bundle: Path, motion: Path, run: Path, output: Path) -> dict:
    """Check physics, solved contacts and actuator forces, without policy inference.

    This proves the recorded commands reproduce the reached states. It does not
    certify where the commands came from or make a skill quality judgement.
    Input files must match their declared hashes, including the full integration
    snapshot, which retains solver warm-start and control state absent from qpos/qvel.
    """
    bundle, motion, run, output = map(Path, (bundle, motion, run, output))
    with open_assessment_inputs(bundle, motion, run, output) as inputs:
        c, result = inputs.contract, inputs.result
        if c["packages"]["mujoco"] != mujoco.__version__:
            raise ValueError("Action replay requires the exported MuJoCo version")
        model = mujoco.MjModel.from_binary_path(str(inputs.model_path))
        validate_model_contract(model, c)
        flags = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        actions = inputs.trajectory.arrays["actions"]
        states = inputs.trajectory.states
        contacts = inputs.trajectory.arrays["physical_contacts"]
        substeps = len(actions) * c["decimation"]
        with TemporaryDirectory(
            prefix=".supermimic-action-replay-", dir=inputs.model_path.parent
        ) as d:
            pinned = Path(d)
            for name, field in (
                ("initialization.npz", "initialization_sha256"),
                ("actuator-forces.npz", "actuator_forces_sha256"),
            ):
                if not isinstance(result.get(field), str):
                    raise ValueError(f"Action replay requires {field}")
                snapshot_file(run / name, pinned / name, expected_sha256=result[field])
            initial = _read_arrays(
                pinned / "initialization.npz",
                {
                    "integration_state": ((mujoco.mj_stateSize(model, flags),), "float64"),
                    "state_flags": ((), "int64"),
                },
            )
            if int(initial["state_flags"]) != flags:
                raise ValueError("Action replay requires mjSTATE_INTEGRATION")
            forces = _read_arrays(
                pinned / "actuator-forces.npz",
                {"actuator_force": ((substeps, model.nu), "float64")},
            )["actuator_force"]
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, initial["integration_state"], flags)
        mujoco.mj_kinematics(model, data)
        floors = {
            i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        }
        feet = [
            {
                i
                for i in range(model.ngeom)
                if model.geom_bodyid[i] == model.body("robot/" + side + "_ankle_roll_link").id
            }
            for side in ("left", "right")
        ]
        default = np.asarray(c["default_joint_pos"], dtype=np.float32)
        scale = np.asarray(c["action_scale"], dtype=np.float32)
        perturbation = InitialPerturbation.from_mapping(result.get("initialization"))
        reference = MotionReference(inputs.motion_path, default)
        declared = initialize_data(model, c, reference, perturbation)
        expected_initial = np.empty(mujoco.mj_stateSize(model, flags))
        mujoco.mj_getState(model, declared, expected_initial, flags)
        initialization_matches = _bitexact(initial["integration_state"], expected_initial)
        checks = {"states": True, "physical_contacts": True, "actuator_forces": True}
        first_mismatch = None

        def compare(field, actual, expected, index):
            nonlocal first_mismatch
            if not _bitexact(actual, expected):
                checks[field] = False
                if first_mismatch is None:
                    first_mismatch = {"field": field, "index": index}

        compare("states", np.r_[data.qpos, data.qvel], states[0], 0)
        for k, action in enumerate(actions):
            target = default + scale * action
            data.ctrl[:] = target[c["actuator_joint_indices"]]
            for sub in range(c["decimation"]):
                mujoco.mj_step(model, data)
                for warning in (
                    mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC,
                ):
                    if data.warning[warning].number:
                        raise ValueError("MuJoCo warning during independent action replay")
                index = k * c["decimation"] + sub
                compare(
                    "physical_contacts",
                    ground_contact_row(model, data, floors, feet),
                    contacts[index],
                    index,
                )
                compare("actuator_forces", data.actuator_force, forces[index], index)
            mujoco.mj_forward(model, data)
            compare("states", np.r_[data.qpos, data.qvel], states[k + 1], k + 1)
        return {
            "passed": all(checks.values()) and initialization_matches,
            "declared_initialization_bitexact": initialization_matches,
            "fields_bitexact": checks,
            "first_mismatch": first_mismatch,
            "controls": len(actions),
            "physics_substeps": substeps,
            "trajectory_sha256": inputs.trajectory.sha256,
            "initialization_sha256": result["initialization_sha256"],
            "actuator_forces_sha256": result["actuator_forces_sha256"],
            "implementation_sha256": sha256(Path(__file__)),
            "scope": "Independent native physics replay of saved commands; no policy inference or skill acceptance.",
        }
