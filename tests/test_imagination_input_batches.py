"""Opt-in CUDA comparison against per-frame input calculations, without physics noise."""

import os
from pathlib import Path

import pytest
import torch

MOTION = Path(os.environ.get("SUPERMIMIC_INPUT_TEST_MOTION", "data/npz/g1/walk1_subject1.npz"))
pytestmark = pytest.mark.skipif(
    os.environ.get("SUPERMIMIC_GPU_INPUT_TESTS") != "1"
    or not torch.cuda.is_available()
    or not MOTION.is_file(),
    reason="opt-in bounded CUDA input-batching comparison with prepared G1 motion",
)


@pytest.mark.parametrize("horizon", [8, 32])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("anchor", [False, True])
def test_input_batches_preserve_rollout_rng_and_actor_gradients(
    monkeypatch, horizon, compact, anchor
):
    import supermimic.imagination.rollout as module
    from supermimic.config import CriticCfg, PolicyCfg, WorldModelCfg
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.motion_library import MotionLibrary
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.imagination.objective import policy_objective
    from supermimic.interfaces import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.policy.nets import Critic, Policy
    from supermimic.world_model.heads import AnalyticHeads, flatten_history_term_major
    from supermimic.world_model.reference import ReferenceFrame
    from supermimic.world_model.world_model import WorldModel

    torch.set_num_threads(1)
    torch.manual_seed(102)
    device, n = "cuda:0", 64
    model = load_g1_model()
    default = torch.tensor(model.qpos0[7:], device=device, dtype=torch.float32)
    lib = MotionLibrary([MOTION], TRACKED_BODIES, default, device, compact=compact)
    heads = AnalyticHeads(
        G1Kinematics(model, device),
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        torch.tensor(model.jnt_range[1:], device=device, dtype=torch.float32),
        default,
    )
    wm = WorldModel(WorldModelCfg(hidden=(32,), term_hidden=(16,)), heads, device)
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.001)
    wm.requires_grad_(False)
    policy = Policy(
        PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,), goal_dim=680 + 9 * anchor)
    ).to(device)
    critic = Critic(CriticCfg(hidden=(16,)), pcfg=policy.cfg).to(device).requires_grad_(False)
    frames = torch.arange(n, device=device) * 10 + 5
    states = torch.stack([lib.ref71(frames + i) for i in range(10)])
    states[..., 39:42] = quat_apply_inverse(states[..., 3:7], states[..., 39:42])
    actions = torch.randn(10, n, 29, device=device) * 0.01
    xi = torch.zeros(n, 33, device=device)
    xi[:, 0] = 0.75
    start = StartBatch(states[-1], xi, frames + 9, actions[-1], states, actions)
    parameters = tuple(policy.parameters())
    individual_observation, goal_window = module.analytic_observation, lib.goal_window
    root_error = heads.root_error_features

    def per_frame_observation(state, previous, context, q0):
        return torch.stack(
            [
                individual_observation(state[i], previous[i], context, q0)
                for i in range(state.shape[0])
            ]
        )

    def per_step_goals(flat_frames, future_frames):
        return torch.cat(
            [goal_window(frame, future_frames) for frame in flat_frames.reshape(horizon + 1, n)]
        )

    def per_step_errors(flat_states, references):
        rows = []
        for t in range(horizon + 1):
            sl = slice(t * n, (t + 1) * n)
            ref = ReferenceFrame(
                None, None, None, None, references.root_pos_w[sl], references.root_quat_w[sl]
            )
            rows.append(root_error(flat_states[sl], ref))
        return torch.cat(rows)

    def objective(rollout):
        alive = rollout.survival[-1].detach() > 1e-6
        feat = wm.spec.canonical(rollout.states[-1])
        value = critic(
            torch.where(alive[:, None], feat, torch.zeros_like(feat)),
            torch.where(alive[:, None], rollout.goals[-1], torch.zeros_like(rollout.goals[-1])),
            wm.spec.normalize_xi(start.xi),
            torch.where(alive[:, None], rollout.e_root[-1], torch.zeros_like(rollout.e_root[-1])),
        )
        return policy_objective(rollout, torch.where(alive, value, torch.zeros_like(value)), 0.99)

    with monkeypatch.context() as reference:
        reference.setattr(
            module,
            "_flatten_history_frames",
            lambda frames, order: flatten_history_term_major(torch.stack(frames)),
        )
        reference.setattr(module, "analytic_observation", per_frame_observation)
        reference.setattr(lib, "goal_window", per_step_goals)
        reference.setattr(heads, "root_error_features", per_step_errors)
        torch.manual_seed(876)
        expected = module.imagine(
            wm, policy, lib, start, horizon, obs_noise=True, anchor_obs=anchor
        )
        expected_gradients = torch.autograd.grad(objective(expected), parameters)
        expected_rng = torch.cuda.get_rng_state()
    torch.manual_seed(876)
    actual = module.imagine(wm, policy, lib, start, horizon, obs_noise=True, anchor_obs=anchor)
    gradients = torch.autograd.grad(objective(actual), parameters)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    for name in ("states", "actions", "rewards", "dhat", "survival", "goals", "e_root", "frames"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=3e-6, rtol=3e-6
        )
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-6, rtol=1e-4)
