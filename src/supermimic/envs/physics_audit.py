"""Fail on MuJoCo Warp capacity truncation instead of accepting altered physics."""

from __future__ import annotations

import torch


def capacity_overflow_mask() -> int:
    from mujoco_warp import OverflowType

    # Reaching a configured solver iteration budget is distinct from truncating
    # contacts/constraints. Preserve that diagnostic, but do not call it capacity loss.
    return int(OverflowType.ALL & ~(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS))


def check_capacity(overflow, mask: int) -> None:
    flags = torch.bitwise_and(overflow, mask)
    if torch.any(flags != 0):
        bad = flags.detach().cpu().tolist()
        raise RuntimeError(f"MuJoCo Warp contact/constraint capacity overflow: {bad}")


def install_capacity_audit(sim) -> None:
    """Check after every physical substep, before an episode reset can clear flags."""
    original = sim.step
    mask = capacity_overflow_mask()

    def audited_step():
        result = original()
        check_capacity(sim.data.overflow, mask)
        return result

    sim.step = audited_step
