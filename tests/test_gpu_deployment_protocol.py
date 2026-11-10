"""CPU deployment inputs must match the real mjlab reset/advance protocol."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.deployment.core import MotionReference, ObservationHistory, gravity

MOTIONS = Path(os.environ.get("SUPERMIMIC_TEST_MOTIONS", "data/npz/validation"))
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MOTIONS.exists(),
    reason="needs CUDA and prepared validation motions",
)


@pytest.mark.parametrize("anchor_obs", [False, True])
def test_deployment_input_clock_and_history_match_mjlab(anchor_obs):
    from supermimic.deployment.contract import build_model_and_contract
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO

    motion = MOTIONS / "walk1_subject1.npz"
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(motion)],
            num_envs=1,
            obs_noise=False,
            reset_noise=False,
            action_noise_std=0,
            domain_randomization=False,
            record_transitions=False,
            anchor_obs=anchor_obs,
        ),
        "cuda:0",
    )
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = 0
        obs, _ = env.reset()
        assert int(cmd.frame[0]) == 1  # pinned mjlab calls command update after reset
        default_q = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
        _, contract = build_model_and_contract()
        action_term = env.action_manager.get_term("joint_pos")
        assert action_term.target_names == contract["joint_names"]
        np.testing.assert_allclose(action_term.scale[0].cpu(), contract["action_scale"], atol=1e-6)
        ref = MotionReference(motion, default_q)
        history = ObservationHistory(10)
        io = StateIO(env)
        previous = np.zeros(29, dtype=np.float32)
        for step in range(4):
            state = io.read()[0].cpu().numpy()
            proprio = np.concatenate(
                [state[7:36] - default_q, state[42:71], state[39:42], gravity(state[3:7]), previous]
            )
            np.testing.assert_allclose(
                history.append(proprio), obs["policy"][0].cpu().numpy(), atol=3e-6
            )
            goal = ref.goal(step + 1, 10)
            if anchor_obs:
                from supermimic.deployment.core import rotation
                from supermimic.world_model.heads import anchor_relative_obs

                robot = env.scene["robot"]
                i = robot.body_names.index("torso_link")
                pos = robot.data.body_link_pos_w[:, i] - env.scene.env_origins
                quat = robot.data.body_link_quat_w[:, i]
                frame = step + 1
                pose = rotation(quat[0].cpu().numpy())
                relative = np.concatenate(
                    [
                        pose.inv().apply(ref.anchor_pos[frame] - pos[0].cpu().numpy()),
                        (pose.inv() * rotation(ref.anchor_quat[frame]))
                        .as_matrix()[:, :2]
                        .reshape(-1),
                    ]
                )
                analytic = anchor_relative_obs(
                    pos,
                    quat,
                    torch.tensor(ref.anchor_pos[frame : frame + 1], device="cuda:0"),
                    torch.tensor(ref.anchor_quat[frame : frame + 1], device="cuda:0"),
                )
                np.testing.assert_allclose(relative, analytic[0].cpu().numpy(), atol=3e-6)
                goal = np.concatenate([goal, relative])
            np.testing.assert_allclose(goal, obs["goal"][0].cpu().numpy(), atol=3e-6)
            # Distinct components expose joint permutations hidden by uniform actions.
            action = torch.linspace(-0.02, 0.03, 29, device="cuda:0")[None] * step
            previous = action[0].cpu().numpy()
            obs, _, term, tout, _ = env.step(action)
            np.testing.assert_allclose(
                action_term._processed_actions[0].cpu().numpy(),
                default_q + np.asarray(contract["action_scale"]) * previous,
                atol=2e-6,
            )
            assert not bool(term[0] | tout[0])
    finally:
        env.close()


def test_benchmark_stops_at_last_frame_and_advances_partial_queue(tmp_path):
    from supermimic.eval.benchmark import run_benchmark
    from supermimic.policy.nets import Policy, PolicyCfg

    clips = []
    with np.load(MOTIONS / "walk1_subject1.npz", allow_pickle=False) as original:
        frames = original["joint_pos"].shape[0]
        for length in (4, 6, 8):
            path = tmp_path / f"walk-{length}.npz"
            arrays = {
                name: original[name][:length]
                if original[name].ndim and original[name].shape[0] == frames
                else original[name]
                for name in original.files
            }
            np.savez(path, **arrays)
            clips.append(str(path))
    policy = Policy(PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,))).cuda().eval()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    results = run_benchmark(policy, clips, num_envs=2, verbose=False)
    assert [r.steps for r in results] == [3, 5, 7]
    assert all(not r.failed and r.coverage == 1 for r in results)
