"""Bounded reward logs retain contiguous time when physical episodes run longer."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand


def command_fixture(num_envs=1):
    command = object.__new__(MultiClipMotionCommand)
    command._env = SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        reward_buf=torch.ones(num_envs),
        termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
    )
    command.cfg = SimpleNamespace(
        gamma=0.5, ema_alpha=1.0, min_remaining_steps=2, completion_aware_sampling=False
    )
    command.max_ep_steps = 4
    command.ep_len = torch.zeros(num_envs, dtype=torch.long)
    command.ep_rewards = torch.zeros(num_envs, 4)
    command.ep_bins = torch.zeros(num_envs, 4, dtype=torch.long)
    command.frame = torch.zeros(num_envs, dtype=torch.long)
    command._gamma_pow = 0.5 ** torch.arange(4, dtype=torch.double)
    command.clip_end = torch.zeros(num_envs, dtype=torch.bool)
    command.library = SimpleNamespace(num_bins=8, bin_of_frame=lambda frame: frame)
    command.bin_return = torch.zeros(8)
    command.metrics = {"episode_return_to_go": torch.zeros(num_envs)}
    command.bootstrap_fn = None
    return command


def test_long_episode_bootstraps_at_chunk_boundary_without_losing_or_retiming_rewards():
    command = command_fixture()
    boundaries = []

    def value_at_boundary(ids):
        boundaries.append(command.frame[ids].tolist())
        # Two rewards remain after the first four controls: 1 + gamma * 1.
        return torch.full((len(ids),), 1.5)

    command.bootstrap_fn = value_at_boundary
    for frame in range(6):
        command.frame[:] = frame
        command._env.termination_manager.terminated[:] = frame == 5
        command._log_step_rewards()
    command._finalize_episodes(torch.tensor([0]))
    assert boundaries == [[3]]
    # Direct discounted sums of the original, uninterrupted six-reward stream.
    assert command.bin_return[:6].tolist() == pytest.approx(
        [1.96875, 1.9375, 1.875, 1.75, 1.5, 1.0]
    )
    assert command.ep_len.tolist() == [0]


@pytest.mark.parametrize("completed", [False, True])
def test_terminal_or_successful_boundary_is_finalized_once_without_bootstrap(completed):
    command = command_fixture()
    command.cfg.completion_aware_sampling = completed
    calls = []
    command.bootstrap_fn = lambda ids: calls.append(ids) or torch.zeros(len(ids))
    for frame in range(4):
        command.frame[:] = frame
        command.clip_end[:] = completed and frame == 3
        command._env.termination_manager.terminated[:] = not completed and frame == 3
        command._log_step_rewards()
    before = command.bin_return.clone()
    command._finalize_episodes(torch.tensor([0]))
    assert not calls and command.ep_len.tolist() == [0]
    assert torch.equal(before, command.bin_return)
    expected = [2, 2, 2, 2] if completed else [1.875, 1.75, 1.5, 1]
    assert before[:4].tolist() == pytest.approx(expected)


def test_without_critic_only_the_uncertain_chunk_tail_is_omitted():
    command = command_fixture()
    for frame in range(4):
        command.frame[:] = frame
        command._log_step_rewards()
    assert command.bin_return[:4].tolist() == pytest.approx([1.875, 1.75, 1.5, 0])
    assert command.ep_len.tolist() == [0]
    command.frame[:] = 4
    command._log_step_rewards()
    assert command.ep_bins[0, 0] == 4 and command.ep_len.tolist() == [1]


def test_masked_chunk_flush_preserves_another_environments_pending_log():
    command = command_fixture(num_envs=2)
    command.bootstrap_fn = lambda ids: torch.full((len(ids),), 2.0)
    for frame in range(2):
        command.frame[:] = frame
        command._log_step_rewards()
    pending = command.ep_rewards[1].clone()
    for frame in range(2, 4):
        command.frame[:] = frame
        command._log_step_rewards(torch.tensor([True, False]))
    assert command.ep_len.tolist() == [0, 2]
    assert torch.equal(command.ep_rewards[1], pending)
    assert command.ep_bins[1, :2].tolist() == [0, 1]
    assert command.bin_return[:4].tolist() == pytest.approx([2, 2, 2, 2])
