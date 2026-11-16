"""Own verified assessment inputs for the complete scoring operation.

Input preparation needs NumPy, not a simulator or policy runtime. Consumers must
use the yielded motion/model paths within this context: both are private,
read-only snapshots removed on every exit, including a scoring failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from supermimic.artifacts import sha256, snapshot_file
from supermimic.data.schema import open_motion, validate_motion
from supermimic.deployment.evidence import (
    read_json_evidence,
    verify_artifact_identity,
    verify_control_dt,
)
from supermimic.deployment.trajectory import (
    Trajectory,
    open_trajectory,
    validate_reference_clock,
    validate_transitions,
)


@dataclass(frozen=True)
class AssessmentInputs:
    result: dict
    contract: dict
    trajectory: Trajectory
    motion_path: Path
    model_path: Path
    body_names: tuple[str, ...]
    positions: np.ndarray
    quaternions: np.ndarray
    motion_frames: int
    evidence: dict

    @property
    def completed_motion(self) -> bool:
        """Keep the skill gate's full-duration/failure rule, separate from admission."""
        return bool(
            self.result["completed_motion"]
            and len(self.trajectory.arrays["actions"]) == self.motion_frames - 1
            and self.result["failure"] is None
        )


@contextmanager
def open_assessment_inputs(
    bundle: Path, motion: Path, run: Path, output: Path
) -> Iterator[AssessmentInputs]:
    """Admit an uninterrupted result and hold its verified inputs through scoring.

    A normal result may describe a fall or incomplete motion; those remain skill
    failures for the judge. Contact row alignment retains the assessor's stricter
    full-control policy, unlike diagnostic plotting of partial physics steps.
    Legacy archives may omit their timestep and contact column labels. Only this
    consumer explicitly supplies the verified contract timestep as a fallback.
    """
    bundle, motion, run, output = map(Path, (bundle, motion, run, output))
    result, result_sha = read_json_evidence(run / "result.json")
    contract, contract_sha = read_json_evidence(bundle / "contract.json")
    if result["status"] != "complete" or result["auto_reset"]:
        raise ValueError("Assessment requires a finished uninterrupted physical rollout")
    required = ("motion_sha256", "trajectory_sha256", "contract_sha256", "checkpoint_sha256")
    if any(not isinstance(result.get(field), str) for field in required):
        raise ValueError("Assessment requires all recorded artifact fingerprints")
    verify_artifact_identity(
        result,
        contract_sha256=contract_sha,
        checkpoint_sha256=contract["checkpoint_sha256"],
    )
    for name, expected in contract["files"].items():
        # The native model is checked while creating the snapshot loaded below.
        if name != "model.mjb" and sha256(bundle / name) != expected:
            raise ValueError(f"Modified bundle file: {name}")
    # Preserve the public assessor's resource admission before copying a source.
    # The owned publication is independently validated after snapshotting it.
    validate_motion(motion)
    existing = output.expanduser().resolve().parent
    while not existing.exists():
        existing = existing.parent
    with TemporaryDirectory(prefix=".supermimic-assessment-", dir=existing) as temporary:
        pinned_motion = Path(temporary) / "motion" / motion.name
        motion_sha = snapshot_file(motion, pinned_motion, expected_sha256=result["motion_sha256"])
        pinned_model = Path(temporary) / "model.mjb"
        model_sha = snapshot_file(
            bundle / "model.mjb", pinned_model, expected_sha256=contract["files"]["model.mjb"]
        )
        with open_motion(pinned_motion) as (data, info):
            names = tuple(data["body_names"].tolist())
            positions, quaternions = data["body_pos_w"], data["body_quat_w"]
            motion_frames = info["frames"]
        with open_trajectory(
            run / "trajectory.npz",
            fields=("actions", "reference_frame", "physical_contacts"),
            fallback_control_dt=contract["control_dt"],
        ) as trajectory:
            identity = verify_artifact_identity(
                result,
                trajectory_sha256=trajectory.sha256,
                motion_sha256=motion_sha,
                contract_sha256=contract_sha,
                checkpoint_sha256=contract["checkpoint_sha256"],
            )
            verify_control_dt(trajectory.control_dt, contract=contract, result=result)
            validate_transitions(trajectory)
            if len(trajectory.arrays["physical_contacts"]) != (
                len(trajectory.arrays["actions"]) * contract["decimation"]
            ):
                raise ValueError("Inconsistent trajectory lengths")
            validate_reference_clock(trajectory, motion_frames, result["reference_goal_offset"])
            yield AssessmentInputs(
                result=result,
                contract=contract,
                trajectory=trajectory,
                motion_path=pinned_motion,
                model_path=pinned_model,
                body_names=names,
                positions=positions,
                quaternions=quaternions,
                motion_frames=motion_frames,
                evidence={
                    "artifact_identity": identity,
                    "contract_sha256": contract_sha,
                    "result_sha256": result_sha,
                    "model_sha256": model_sha,
                    "control_dt": trajectory.control_dt,
                    "control_dt_source": trajectory.control_dt_source,
                    "contact_layout_verified": trajectory.contact_layout_verified,
                    "reference_clock": {
                        "source": "result.json",
                        "offset": result["reference_goal_offset"],
                    },
                },
            )
