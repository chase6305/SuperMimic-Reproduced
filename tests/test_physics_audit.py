from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.physics_audit import (
    capacity_overflow_mask,
    check_capacity,
    install_capacity_audit,
)


def test_solver_iteration_budget_is_not_capacity_truncation():
    from mujoco_warp import OverflowType

    mask = capacity_overflow_mask()
    check_capacity(torch.tensor([int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)]), mask)
    for value in [OverflowType.NEFC, OverflowType.NARROWPHASE, OverflowType.BROADPHASE]:
        with pytest.raises(RuntimeError, match="capacity overflow"):
            check_capacity(torch.tensor([0, int(value)]), mask)


def test_capacity_audit_observes_each_physical_step():
    from mujoco_warp import OverflowType

    sim = SimpleNamespace(data=SimpleNamespace(overflow=torch.zeros(2, dtype=torch.int32)))
    calls = []

    def physical_step():
        calls.append(1)
        sim.data.overflow[1] = int(OverflowType.NARROWPHASE)

    sim.step = physical_step
    install_capacity_audit(sim)
    with pytest.raises(RuntimeError, match="capacity overflow"):
        sim.step()
    assert len(calls) == 1
