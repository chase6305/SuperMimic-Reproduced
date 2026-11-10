"""Dense tracking gradients must target the commanded reached state safely."""

import torch

from supermimic.imagination.tracking import tracking_auxiliary


def test_dense_tracking_penalty_has_correct_direction_and_masks_dead_branches():
    predicted = torch.zeros(2, 2, 71)
    predicted[..., 3] = 1
    target = predicted.clone()
    target[0, 0, 7:36] = 0.4
    target[0, 0, 2] = 0.1
    target[0, 0, 38] = 1
    predicted[1, 1] = torch.inf
    predicted.requires_grad_()
    survival = torch.tensor([[1.0, 1.0], [1.0, 0.0]], requires_grad=True)
    loss, terms = tracking_auxiliary(predicted, target, survival, (5.0, 10.0, 2.0, 0.5))
    assert torch.isfinite(loss) and torch.isfinite(terms).all()
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert (predicted.grad[0, 0, 7:36] < 0).all()
    assert predicted.grad[0, 0, 2] < 0 and predicted.grad[0, 0, 38] < 0
    assert predicted.grad[1, 1].abs().sum() == 0
    assert survival.grad is None
    exact = target.detach().requires_grad_()
    zero, _ = tracking_auxiliary(exact, target.detach(), survival, (5.0, 10.0, 2.0, 0.5))
    assert zero == 0
    zero.backward()
    assert torch.isfinite(exact.grad).all() and exact.grad.abs().sum() == 0


def test_geodesic_orientation_supplies_half_turn_gradient_and_masks_dead_states():
    predicted = torch.zeros(1, 2, 71)
    predicted[0, 0, 6] = 1
    predicted[0, 1] = torch.inf
    target = torch.zeros_like(predicted)
    target[..., 3] = 1
    survival = torch.tensor([[1.0, 0.0]])
    predicted.requires_grad_()
    chordal, _ = tracking_auxiliary(predicted, target, survival, (0.0, 0.0, 1.0, 0.0), dt=1)
    chordal.backward()
    assert predicted.grad[0, 0, 3] == 0
    predicted.grad = None
    geodesic, _ = tracking_auxiliary(
        predicted, target, survival, (0.0, 0.0, 1.0, 0.0), dt=1, orientation_mode="geodesic"
    )
    assert abs(float(geodesic.detach()) - torch.pi**2) < 1e-5
    geodesic.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0, 1].abs().sum() == 0
    assert predicted.grad[0, 0, 3].abs() > 1


def test_angular_velocity_penalty_uses_world_reference_and_body_state_frames():
    predicted = torch.zeros(1, 2, 71)
    predicted[0, 0, 3] = predicted[0, 0, 6] = 2**-0.5  # yaw +90 degrees
    predicted[0, 0, 40] = -1  # body -y is world +x
    target = predicted.clone()
    target[0, 0, 39:42] = torch.tensor([1.0, 0.0, 0.0])
    predicted[0, 1] = torch.inf
    survival = torch.tensor([[1.0, 0.0]])
    weights = (0.0, 0.0, 0.0, 0.0)
    exact, terms = tracking_auxiliary(
        predicted, target, survival, weights, angular_velocity_weight=0.1
    )
    assert exact < 1e-12 and terms.shape == (5,)
    predicted[0, 0, 40] = -0.5
    predicted.requires_grad_()
    loss, _ = tracking_auxiliary(predicted, target, survival, weights, angular_velocity_weight=0.1)
    loss.backward()
    assert loss > 0 and predicted.grad[0, 0, 40] > 0
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0, 1].abs().sum() == 0
