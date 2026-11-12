"""Export a CPU actor plus a hashed model and explicit control/observation ABI."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.artifacts import sha256, write_json
from supermimic.checkpoints import policy_from_checkpoint, read_checkpoint
from supermimic.deployment.contract import build_model_and_contract
from supermimic.interfaces import observation_dim


def export_policy(checkpoint: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    loaded = read_checkpoint(checkpoint, fingerprint=True)
    ck = loaded.data
    cfg = ck["config"]
    actor = policy_from_checkpoint(ck)
    pcfg = actor.cfg
    model, contract = build_model_and_contract(
        pcfg.history, cfg["env"]["future_frames"], cfg["env"].get("anchor_obs", False)
    )
    control = ck.get("control_contract")
    if control is not None:
        if control["joint_names"] != contract["joint_names"]:
            raise ValueError("Checkpoint joint order differs from the export model")
        for key in ("default_joint_pos", "action_scale", "control_dt"):
            if not np.allclose(control[key], contract[key], atol=1e-6, rtol=1e-6):
                raise ValueError(f"Checkpoint {key} differs from the export model")
    contract["checkpoint_control_contract_recorded"] = control is not None
    contract["action_clipping"] = (
        None
        if pcfg.action_clip is None
        else {
            "normalized_min": -pcfg.action_clip,
            "normalized_max": pcfg.action_clip,
            "location": "embedded in policy.ts; observation history uses clipped clean output",
        }
    )
    obs = torch.zeros(1, observation_dim(pcfg.history))
    goal = torch.zeros(1, pcfg.goal_dim)
    traced = torch.jit.freeze(torch.jit.trace(actor, (obs, goal)).eval())
    # Compare independent nonzero inputs, including a different batch size.
    generator = torch.Generator().manual_seed(2026)
    worst = 0.0
    with torch.no_grad():
        for n in (1, 4, 2):
            o = torch.randn(n, obs.shape[1], generator=generator)
            g = torch.randn(n, goal.shape[1], generator=generator)
            reference, actual = actor(o, g), traced(o, g)
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
            worst = max(worst, float((actual - reference).abs().max()))
    output.mkdir(parents=True, exist_ok=False)
    traced.save(str(output / "policy.ts"))
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    licenses = Path(__file__).parents[1] / "licenses"
    for path in licenses.glob("*.txt"):
        shutil.copyfile(path, output / path.name)
    (output / "NOTICE.txt").write_text(
        "Independent SuperMimic reproduction (MIT). G1 scene from the pinned mjlab "
        "dependency (Apache-2.0), with Unitree robot assets (BSD-3-Clause). "
        "Model/data provenance is recorded in contract.json and the originating run.\n"
    )
    contract.update(
        checkpoint_sha256=loaded.sha256,
        checkpoint_iteration=int(ck.get("iteration", 0)),
        packages={name: importlib.metadata.version(name) for name in ("torch", "mujoco", "mjlab")},
        export_parity_max_abs_error=worst,
        policy_input_normalization="Fixed observation/goal scaling is embedded in policy.ts; supply unscaled contract features",
        training_metadata={
            "seed": cfg.get("seed"),
            "motion_files": ck.get("motion_files"),
            "world_model_hidden": cfg.get("world_model", {}).get("hidden"),
            "policy_encoder_hidden": list(pcfg.encoder_hidden),
            "policy_decoder_hidden": list(pcfg.decoder_hidden),
            "min_imagined_survival": cfg.get("policy_trainer", {}).get("min_survival"),
            "anchor_adaptation": ck.get("anchor_adaptation"),
            "transfer_provenance": ck.get("transfer_provenance"),
            "action_clip": pcfg.action_clip,
            "completion_aware_sampling": cfg.get("env", {}).get("completion_aware_sampling", False),
            "sampling_priority": {
                "windows_s": cfg.get("env", {}).get("priority_windows_s", []),
                "mixture_share": cfg.get("env", {}).get("priority_reset_ratio", 0.0),
            },
            "wm_one_step_batches_per_update": cfg.get("wm_trainer", {}).get(
                "one_step_batches_per_update", 0
            ),
            "world_model_fitting_window": cfg.get("wm_trainer", {}).get("window", 8),
            "policy_gradient_accumulation_steps": cfg.get("policy_trainer", {}).get(
                "gradient_accumulation_steps", 1
            ),
            "imagination_reset_start_fraction": cfg.get("policy_trainer", {}).get(
                "reset_start_fraction", 0.0
            ),
            "tracking_objective": cfg.get("policy_trainer", {}).get(
                "tracking_objective", "baseline"
            ),
            "policy_update_scope": cfg.get("policy_trainer", {}).get("update_scope", "all"),
            "imagination_use_critic": cfg.get("policy_trainer", {}).get("use_critic", True),
            "imagination_horizon": cfg.get("policy_trainer", {}).get("horizon"),
            "training_environment": {
                name: cfg.get("env", {}).get(name)
                for name in ("obs_noise", "reset_noise", "domain_randomization", "action_noise_std")
            },
            "tracking_auxiliary": {
                name: cfg.get("policy_trainer", {}).get(
                    name,
                    (1.0,) * 4
                    if name == "effector_position_weights"
                    else 0.0
                    if name == "root_xy_tracking_weight"
                    else None,
                )
                for name in (
                    "joint_tracking_weight",
                    "root_xy_tracking_weight",
                    "height_tracking_weight",
                    "orientation_tracking_weight",
                    "vertical_velocity_tracking_weight",
                    "angular_velocity_tracking_weight",
                    "foot_support_height_weight",
                    "foot_support_velocity_weight",
                    "effector_position_weight",
                    "effector_position_weights",
                    "orientation_loss",
                    "joint_tracking_weights",
                )
            },
        },
        mjlab_git_commit=json.loads(
            importlib.metadata.distribution("mjlab").read_text("direct_url.json") or "{}"
        )
        .get("vcs_info", {})
        .get("commit_id"),
        files={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    )
    write_json(output / "contract.json", contract)
    return contract
