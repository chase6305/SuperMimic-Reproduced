"""History gathering preserves term order and derivatives for arbitrary strides."""

import pytest
import torch

from supermimic.imagination.rollout import _flatten_history_frames, _history_order
from supermimic.world_model.heads import flatten_history_term_major


@pytest.mark.parametrize("length", [1, 3, 10])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("strided", [False, True])
def test_history_gather_matches_term_slices_and_their_adjoints(length, batch, strided):
    torch.manual_seed(591)
    history = torch.randn(length, batch, 186 if strided else 93, dtype=torch.float64)
    if strided:
        history = history[..., ::2]
    history.requires_grad_()
    actual = _flatten_history_frames(list(history.unbind(0)), _history_order(length, "cpu"))
    expected = flatten_history_term_major(history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(expected)
    actual_gradient = torch.autograd.grad((actual * probe).sum(), history)[0]
    expected_gradient = torch.autograd.grad((expected * probe).sum(), history)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
