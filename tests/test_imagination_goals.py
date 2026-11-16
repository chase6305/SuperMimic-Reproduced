"""Batched lookahead keeps every clip boundary and state-dependent anchor term."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.motion_library import MotionLibrary, goal_features
from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.rollout import imagine
from supermimic.world_model.heads import analytic_observation, anchor_relative_obs


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("anchor", [False, True])
def test_batched_goals_match_individual_lookahead_at_clip_edges(compact, anchor):
    torch.manual_seed(78)
    library = MotionLibrary.__new__(MotionLibrary)
    library.device, library.dtype, library.compact = torch.device("cpu"), torch.float32, compact
    library.default_joint_pos = torch.randn(29) * 0.1
    reference = torch.randn(16, 71) * 0.1
    reference[:, 2] = 0.78
    reference[:, 3:7] /= reference[:, 3:7].norm(dim=-1, keepdim=True)
    library._ref = reference.half() if compact else reference
    library.frame_clip = torch.tensor([0] * 5 + [1] * 11)
    library.clip_last = torch.tensor([4, 15])
    if not compact:
        library.goal_feat = goal_features(library._split(reference), library.default_joint_pos)

    def bodies(frame):
        ref = library.ref(frame)
        return SimpleNamespace(
            pos_w=ref.root_pos_w[:, None],
            quat_w=ref.root_quat_w[:, None],
            lin_vel_w=ref.root_lin_vel_w[:, None],
            ang_vel_w=ref.root_ang_vel_w[:, None],
        )

    library.bodies = bodies
    frames = torch.tensor([0, 3, 4, 6, 13, 15])
    n = len(frames)
    state = reference[frames].clone()
    context = torch.zeros(n, 33)
    action = torch.zeros(n, 29)
    start = StartBatch(
        state, context, frames, action, state[None].repeat(10, 1, 1), action[None].repeat(10, 1, 1)
    )
    heads = SimpleNamespace(
        anchor_index=0,
        root_error_features=lambda s, ref: torch.zeros(len(s), 4),
        robot_bodies=lambda s: {"pos": s[:, None, :3], "quat": s[:, None, 3:7]},
    )

    class Policy(torch.nn.Module):
        def forward(self, observation, goal):
            return torch.zeros(n, 29)

    def step(s, a, previous, xi, ref):
        reached = s.clone()
        reached[:, 0] += 0.04
        return (
            reached,
            torch.zeros(n),
            torch.zeros(8, n),
            torch.full((n,), -100.0),
            analytic_observation(reached, a, xi, library.default_joint_pos),
            reached[:, :7],
        )

    original_goal = library.goal_window
    requests = []

    def record(frame, future_frames):
        requests.append(frame.clone())
        return original_goal(frame, future_frames)

    library.goal_window = record
    rollout = imagine(
        SimpleNamespace(heads=heads),
        Policy(),
        library,
        start,
        horizon=8,
        future_frames=3,
        obs_noise=False,
        anchor_obs=anchor,
        step_fn=step,
    )
    expected_frames = torch.minimum(
        frames[None] + torch.arange(9)[:, None], library.last_frame_of(frames)[None]
    )
    assert torch.equal(rollout.frames, expected_frames)
    for t, frame in enumerate(expected_frames):
        expected = original_goal(frame, 3)
        if anchor:
            ref = library.ref(frame)
            relative = anchor_relative_obs(
                rollout.states[t, :, :3], rollout.states[t, :, 3:7], ref.root_pos_w, ref.root_quat_w
            )
            expected = torch.cat([expected, relative], dim=-1)
        # Compact goals recompute rotations: CPU vector math at a different
        # batch size can differ by a few float32 ulps (rather than a reindexing).
        torch.testing.assert_close(
            rollout.goals[t],
            expected,
            rtol=1e-6 if compact else 0,
            atol=1e-7 if compact else 0,
        )
    assert len(requests) == 1
    assert torch.equal(requests[0], expected_frames.reshape(-1))
