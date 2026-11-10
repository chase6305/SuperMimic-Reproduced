"""Audit an actual CPU MuJoCo rollout against the fixed project skill gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
        import mujoco
        import numpy as np
        from scipy.spatial.transform import Rotation

        import supermimic.eval.acceptance as acceptance_module
        import supermimic.eval.quality as quality_module
        from supermimic.artifacts import sha256, write_json
        from supermimic.data.schema import validate_motion
        from supermimic.deployment.metrics import tracking_metrics
        from supermimic.eval.acceptance import flight_alignment, judge_skill

        result = json.loads((args.run / "result.json").read_text())
        contract = json.loads((args.bundle / "contract.json").read_text())
        trajectory = args.run / "trajectory.npz"
        if result["status"] != "complete" or result["auto_reset"]:
            raise ValueError("Assessment requires a finished uninterrupted physical rollout")
        for actual, expected in (
            (sha256(args.motion), result["motion_sha256"]),
            (sha256(trajectory), result["trajectory_sha256"]),
            (sha256(args.bundle / "contract.json"), result["contract_sha256"]),
            (contract["checkpoint_sha256"], result["checkpoint_sha256"]),
        ):
            if actual != expected:
                raise ValueError("Artifact identity mismatch")
        for name, expected in contract["files"].items():
            if sha256(args.bundle / name) != expected:
                raise ValueError("Modified bundle file")
        validate_motion(args.motion)
        with np.load(trajectory, allow_pickle=False) as d:
            states, actions, frames, contacts = (
                d[k].copy() for k in ["states", "actions", "reference_frame", "physical_contacts"]
            )
        if (
            len(states) != len(actions) + 1
            or len(frames) != len(actions)
            or len(contacts) != len(actions) * contract["decimation"]
        ):
            raise ValueError("Inconsistent trajectory lengths")
        with np.load(args.motion, allow_pickle=False) as d:
            names = d["body_names"].tolist()
            positions = d["body_pos_w"]
            quaternions = d["body_quat_w"]
            n = len(d["joint_pos"])
            if not np.array_equal(
                frames, np.minimum(np.arange(len(frames)) + result["reference_goal_offset"], n - 1)
            ):
                raise ValueError("Reference clock discontinuity")
            model = mujoco.MjModel.from_binary_path(str(args.bundle / "model.mjb"))
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
        metrics = tracking_metrics(states, actions, frames, args.motion, contract)
        flight = flight_alignment(
            np.stack(soles, -1), contacts, contract["control_dt"], contract["physics_dt"]
        )
        complete = (
            result["completed_motion"] and len(actions) == n - 1 and result["failure"] is None
        )
        judgement = judge_skill(metrics, complete, args.skill, flight)
        write_json(
            args.output,
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "trajectory_sha256": result["trajectory_sha256"],
                "motion_sha256": result["motion_sha256"],
                "assessment_implementation_sha256": sha256(Path(__file__)),
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
