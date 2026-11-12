"""Audit an actual CPU MuJoCo rollout against the fixed project skill gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _score_inputs(inputs, skill):
    """Run the existing physical scoring formulas on context-owned inputs."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    from supermimic.deployment.metrics import tracking_metrics
    from supermimic.eval.acceptance import flight_alignment, judge_skill

    contract = inputs.contract
    states = inputs.trajectory.states
    actions = inputs.trajectory.arrays["actions"]
    frames = inputs.trajectory.arrays["reference_frame"]
    contacts = inputs.trajectory.arrays["physical_contacts"]
    names, positions, quaternions, n = (
        inputs.body_names,
        inputs.positions,
        inputs.quaternions,
        inputs.motion_frames,
    )
    model = mujoco.MjModel.from_binary_path(str(inputs.model_path))
    soles = []
    for side in ("left", "right"):
        name = side + "_ankle_roll_link"
        idx = names.index(name)
        body = model.body("robot/" + name).id
        rotation = Rotation.from_quat(quaternions[:, idx][:, [1, 2, 3, 0]])
        bottoms = []
        for i in range(model.ngeom):
            if model.geom_bodyid[i] != body or not model.geom_contype[i]:
                continue
            center = (
                positions[:, idx, 2]
                + rotation.apply(np.broadcast_to(model.geom_pos[i], (n, 3)))[:, 2]
            )
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                fixed = Rotation.from_quat(model.geom_quat[i][[1, 2, 3, 0]])
                axis = (rotation * fixed).apply(np.broadcast_to([0, 0, 1], (n, 3)))[:, 2]
                extent = model.geom_size[i, 0] + model.geom_size[i, 1] * np.abs(axis)
            elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = model.geom_size[i, 0]
            else:
                raise ValueError("Unsupported foot collision geometry")
            bottoms.append(center - extent)
        if not bottoms:
            raise ValueError("No physical foot colliders")
        soles.append(np.min(bottoms, axis=0))
    metrics = tracking_metrics(states, actions, frames, inputs.motion_path, contract)
    flight = flight_alignment(
        np.stack(soles, -1), contacts, contract["control_dt"], contract["physics_dt"]
    )
    judgement = judge_skill(metrics, inputs.completed_motion, skill, flight)
    return metrics, flight, judgement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--skill", choices=("jump", "dance", "fight"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--max-rss-gib",
        type=float,
        default=2.0,
        help="CPU process-tree RSS ceiling, 0.25–2 GiB; system reserves remain unchanged",
    )
    args = p.parse_args()
    if not math.isfinite(args.max_rss_gib) or not 0.25 <= args.max_rss_gib <= 2.0:
        p.error("--max-rss-gib must be finite and within [0.25, 2]")
    from supermimic.cpu_resources import cpu_guard
    from supermimic.resources import ResourceLimits

    with cpu_guard(args.output, limits=ResourceLimits(max_rss_gib=args.max_rss_gib)):
        import supermimic.deployment.assessment as assessment_module
        import supermimic.eval.acceptance as acceptance_module
        import supermimic.eval.quality as quality_module
        from supermimic.artifacts import sha256, write_json
        from supermimic.deployment.assessment import open_assessment_inputs
        from supermimic.runtime import source_manifest

        with open_assessment_inputs(args.bundle, args.motion, args.run, args.output) as inputs:
            metrics, flight, judgement = _score_inputs(inputs, args.skill)
            write_json(
                args.output,
                {
                    "checkpoint_sha256": inputs.result["checkpoint_sha256"],
                    "trajectory_sha256": inputs.trajectory.sha256,
                    "motion_sha256": inputs.result["motion_sha256"],
                    **inputs.evidence,
                    "assessment_implementation_sha256": sha256(Path(__file__)),
                    "assessment_inputs_implementation_sha256": sha256(
                        Path(assessment_module.__file__)
                    ),
                    "assessment_implementation_package_sha256": source_manifest(),
                    "acceptance_implementation_sha256": sha256(Path(acceptance_module.__file__)),
                    "quality_implementation_sha256": sha256(Path(quality_module.__file__)),
                    "judgement": judgement,
                    "tracking": metrics,
                    "flight": flight,
                },
            )
        print(
            json.dumps(
                {
                    "skill": args.skill,
                    "passed": judgement["passed"],
                    "failed_checks": [c["name"] for c in judgement["checks"] if not c["passed"]],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
