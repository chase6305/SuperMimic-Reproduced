"""Successful short clip endings must not monopolize difficult-pose sampling."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand


def command_fixture(enabled):
    # Same rewards, but success, failure, time limit, and simultaneous failure/end.
    command = object.__new__(MultiClipMotionCommand)
    command._env = SimpleNamespace(
        device="cpu",
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, True, False, True])),
    )
    command.cfg = SimpleNamespace(
        gamma=0.99, ema_alpha=1.0, min_remaining_steps=2, completion_aware_sampling=enabled
    )
    command.ep_len = torch.tensor([2, 2, 2, 2])
    command.ep_rewards = torch.full((4, 2), 0.05)
    command.ep_bins = torch.arange(4)[:, None].expand(4, 2)
    command._gamma_pow = torch.tensor([1, 0.99], dtype=torch.double)
    command.clip_end = torch.tensor([True, False, False, True])
    command.bootstrap_fn = None
    command.library = SimpleNamespace(num_bins=4)
    command.bin_return = torch.full((4,), 10.0)
    command.metrics = {"episode_return_to_go": torch.zeros(4)}
    return command


def test_completion_correction_preserves_failures_and_time_limits():
    command = command_fixture(True)
    command._finalize_episodes(torch.arange(4))
    # A 0.05 step reward normalized to the infinite discounted horizon is 5.
    assert command.bin_return[0] == pytest.approx(5.0)
    assert command.bin_return[1] == pytest.approx(0.07475)
    assert command.bin_return[2] == pytest.approx(0.0995)
    assert command.bin_return[3] == pytest.approx(0.07475)
    assert command.ep_len.tolist() == [0, 0, 0, 0]


def test_baseline_discounted_sampler_is_unchanged():
    command = command_fixture(False)
    command._finalize_episodes(torch.arange(4))
    assert command.bin_return.tolist() == pytest.approx([0.07475, 0.07475, 0.0995, 0.07475])
