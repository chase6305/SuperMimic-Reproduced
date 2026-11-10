"""Difficult reset bins must supplement, never replace, full-clip sampling."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.envs.mdp.sampling import priority_distribution, validate_priorities


def library():
    return SimpleNamespace(
        num_clips=1,
        num_frames=500,
        num_bins=10,
        bin_start=torch.arange(0, 500, 50),
        bin_end=torch.arange(50, 501, 50),
        clip_start=torch.tensor([0]),
        frame_clip=torch.zeros(500, dtype=torch.long),
        last_frame_of=lambda frames: torch.full_like(frames, 499),
    )


def test_priority_mixture_preserves_all_bins_and_expected_mass():
    lib = library()
    command = MultiClipMotionCommand.__new__(MultiClipMotionCommand)
    command._env = SimpleNamespace(device="cpu")
    command.library = lib
    command.cfg = SimpleNamespace(
        sampling_mode="return",
        softmax_temp=25,
        g_max=10,
        memory_alpha=0.01,
        uniform_ratio=0.5,
        hard_ratio=0.4,
        memory_ratio=0.1,
        priority_reset_ratio=0.25,
    )
    command.priority_distribution = priority_distribution(lib, ((2, 4),), 0.25)
    command.bin_return = torch.zeros(10)
    command.bin_memory = torch.full((10,), 0.1)
    command.metrics = {"sampling_entropy": torch.zeros(1)}
    torch.manual_seed(47)
    frames = command._sample_frames(20000)
    assert (frames >= 0).all() and (frames <= 497).all()
    assert len(torch.unique(frames // 50)) == 10
    # Baseline uniform mass .2, plus .25 reserved for the two priority bins.
    mass = ((frames >= 100) & (frames < 200)).float().mean().item()
    assert mass == pytest.approx(0.75 * 0.2 + 0.25, abs=0.015)
    assert 0 < command.metrics["sampling_entropy"].item() < 1
    assert priority_distribution(lib, (), 0) is None


def test_priority_windows_reject_invalid_or_wrong_clip_clocks():
    for windows, ratio in [
        ((), 0.25),
        (((2, 1),), 0.25),
        (((-1, 2),), 0.25),
        (((1, float("nan")),), 0.25),
        (((1, 2),), 0.51),
    ]:
        with pytest.raises(ValueError):
            validate_priorities(windows, ratio)
    lib = library()
    with pytest.raises(ValueError, match="duration"):
        priority_distribution(lib, ((9, 11),), 0.25)
    lib.num_clips = 2
    with pytest.raises(ValueError, match="exactly one"):
        priority_distribution(lib, ((2, 4),), 0.25)
