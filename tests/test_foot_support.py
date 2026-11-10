"""Support gradients use the real collider bottoms, including rotating capsules."""

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.imagination.foot_support import FootSupport
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_rotvec


def physical_soles(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    result = []
    for side in ("left", "right"):
        bottoms = []
        for geom in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").split("/")[-1]
            if not name.startswith(side + "_foot") or not model.geom_contype[geom]:
                continue
            extent = model.geom_size[geom, 0]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent += model.geom_size[geom, 1] * abs(data.geom_xmat[geom].reshape(3, 3)[2, 2])
            else:
                assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
            bottoms.append(data.geom_xpos[geom, 2] - extent)
        result.append(min(bottoms))
    return np.array(result)


def test_collision_heights_and_normal_velocities_match_mujoco():
    from supermimic.deployment.contract import build_model_and_contract

    model, _ = build_model_and_contract()
    fk = G1Kinematics(load_g1_model(), dtype=torch.float64)
    support = FootSupport(model, fk)
    rng = np.random.default_rng(5)
    for _ in range(3):
        qpos = model.qpos0.copy()
        qpos[:3] = [0.2, -0.1, 1.0]
        qpos[3:7] = quat_from_rotvec(torch.tensor(rng.normal(0, 0.2, 3))).numpy()
        qpos[7:] = rng.normal(0, 0.15, 29)
        qvel = rng.normal(0, 0.3, model.nv)
        height, velocity = support.soles(fk(torch.tensor(qpos), torch.tensor(qvel)))
        np.testing.assert_allclose(height.numpy(), physical_soles(model, qpos), atol=1e-10)
        plus, minus = qpos.copy(), qpos.copy()
        epsilon = 1e-6
        mujoco.mj_integratePos(model, plus, qvel, epsilon)
        mujoco.mj_integratePos(model, minus, qvel, -epsilon)
        derivative = (physical_soles(model, plus) - physical_soles(model, minus)) / (2 * epsilon)
        np.testing.assert_allclose(velocity.numpy(), derivative, atol=1e-7)


def test_support_loss_pushes_hovering_feet_down_and_masks_dead_branches():
    model = load_g1_model()
    fk = G1Kinematics(model, dtype=torch.float64)
    support = FootSupport(model, fk)
    target = torch.zeros(1, 2, 71, dtype=torch.float64)
    target[..., :36] = torch.tensor(model.qpos0)
    height, _ = support.soles(fk(target[..., :36]))
    target[..., 2] -= height.max()
    predicted = target.clone()
    predicted[0, 0, 2] += 0.08
    predicted[0, 0, 38] = 0.4
    predicted[0, 1] = torch.inf
    predicted.requires_grad_()
    survival = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss, terms = support.loss(predicted, target, survival, 500, 0.5)
    loss.backward()
    assert loss > 0 and torch.isfinite(terms).all()
    assert predicted.grad[0, 0, 2] > 0 and predicted.grad[0, 0, 38] > 0
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0, 1].abs().sum() == 0
    assert survival.grad is None


def test_reference_flight_has_no_stance_penalty():
    model = load_g1_model()
    fk = G1Kinematics(model)
    support = FootSupport(model, fk)
    target = torch.zeros(1, 1, 71)
    target[..., :36] = torch.tensor(model.qpos0, dtype=torch.float32)
    target[..., 2] += 2
    predicted = target.clone().requires_grad_()
    loss, _ = support.loss(predicted, target, torch.ones(1, 1), 500, 0.5)
    loss.backward()
    assert loss == 0 and predicted.grad.abs().sum() == 0


def test_effector_positions_share_actual_torso_xy_and_yaw():
    model = load_g1_model()
    fk = G1Kinematics(model, dtype=torch.float64)
    support = FootSupport(model, fk)
    target = torch.zeros(1, 1, 71, dtype=torch.float64)
    target[..., :36] = torch.tensor(model.qpos0)
    target[..., 2] = 1
    predicted = target.clone()
    predicted[..., :2] += torch.tensor([2.0, -3.0])
    predicted[..., 3:7] = quat_from_rotvec(torch.tensor([0.0, 0.0, 1.4], dtype=torch.float64))
    predicted.requires_grad_()
    loss, terms = support.loss(
        predicted, target, torch.ones(1, 1), 0, 0, effector_position_weight=10
    )
    loss.backward()
    assert terms.shape == (3,) and terms[2] < 1e-12
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[..., :2].abs().max() < 1e-10


def test_effector_gradient_corrects_arm_pose_and_masks_dead_states():
    model = load_g1_model()
    fk = G1Kinematics(model, dtype=torch.float64)
    support = FootSupport(model, fk)
    target = torch.zeros(1, 2, 71, dtype=torch.float64)
    target[..., :36] = torch.tensor(model.qpos0)
    target[..., 2] = 1
    predicted = target.clone()
    shoulder = model.jnt_qposadr[model.joint("left_shoulder_pitch_joint").id]
    predicted[0, 0, shoulder] += 0.2
    predicted[0, 1] = torch.inf
    predicted.requires_grad_()
    survival = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss, terms = support.loss(predicted, target, survival, 0, 0, effector_position_weight=10)
    loss.backward()
    assert loss > 0 and terms[2] > 0 and predicted.grad[0, 0, shoulder] > 0
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0, 1].abs().sum() == 0
    assert survival.grad is None
