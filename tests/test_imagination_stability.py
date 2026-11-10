"""Dead imagined branches must not overflow or poison reachable value targets."""

from types import SimpleNamespace

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.objective import policy_objective, survival_weighted_mse
from supermimic.imagination.rollout import imagine


def test_unreachable_infinite_value_target_has_zero_loss_and_gradient():
    prediction = torch.tensor([1.0, float("inf")], requires_grad=True)
    target = torch.tensor([2.0, float("inf")])
    loss = survival_weighted_mse(prediction, target, torch.tensor([1.0, 0.0]))
    assert loss == 1
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 0.0]))


def test_value_fit_matches_reachability_weighted_expected_loss():
    prediction = torch.tensor([3.0, 4.0], requires_grad=True)
    target = torch.tensor([2.0, 8.0])
    loss = survival_weighted_mse(prediction, target, torch.tensor([1.0, 0.25]))
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.6, -1.6]))


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, observation, goal):
        return self.bias.expand(observation.shape[0], 29)


class ToyLibrary:
    default_joint_pos = torch.zeros(29)

    def last_frame_of(self, frame):
        return torch.full_like(frame, 1000)

    def goal_window(self, frame, future_frames):
        return torch.zeros(len(frame), 68 * future_frames)

    def ref(self, frame):
        quat = torch.zeros(len(frame), 4)
        quat[:, 0] = 1
        return SimpleNamespace(root_pos_w=torch.zeros(len(frame), 3), root_quat_w=quat)

    def bodies(self, frame):
        reference = self.ref(frame)
        return SimpleNamespace(
            pos_w=reference.root_pos_w[:, None],
            quat_w=reference.root_quat_w[:, None],
            lin_vel_w=torch.zeros(len(frame), 1, 3),
            ang_vel_w=torch.zeros(len(frame), 1, 3),
        )


def test_dead_feedback_branch_is_absorbing_and_has_no_future_policy_gradient():
    state = torch.zeros(2, 71)
    state[:, 0] = state[:, 3] = 1
    start = StartBatch(
        state,
        torch.zeros(2, 33),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, 29),
        state[None].repeat(10, 1, 1),
        torch.zeros(10, 2, 29),
    )
    heads = SimpleNamespace(anchor_index=0, root_error_features=lambda s, r: torch.zeros(len(s), 4))
    policy = ToyPolicy()
    calls = []

    def exploding_step(current, action, previous, context, reference):
        calls.append(current.detach().clone())
        following = current.clone()
        following[:, 0] = current[:, 0] * 10 + action[:, 0]
        reward = following[:, 0]
        return (
            following,
            reward,
            reward[None].repeat(8, 1),
            torch.full((2,), 100.0),
            torch.zeros(2, 93),
            following[:, :7],
        )

    rollout = imagine(
        SimpleNamespace(heads=heads),
        policy,
        ToyLibrary(),
        start,
        horizon=64,
        obs_noise=False,
        step_fn=exploding_step,
    )
    assert torch.isfinite(rollout.states).all()
    assert torch.isfinite(rollout.rewards).all()
    assert not rollout.survival[1:].any()
    assert not rollout.rewards[1:].any()
    for supplied in calls[1:]:
        torch.testing.assert_close(supplied, state)
    objective = policy_objective(rollout, None, gamma=0.99)
    objective.backward()
    torch.testing.assert_close(policy.bias.grad, torch.tensor(1.0))
