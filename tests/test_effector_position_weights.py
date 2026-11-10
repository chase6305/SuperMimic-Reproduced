"""Per-limb training coefficients preserve legacy defaults and physical loss gradients."""

import json
from dataclasses import asdict

import pytest
import torch

from supermimic.cli import parser
from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.imagination.foot_support import FootSupport
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.profiles import training_config


@pytest.fixture(scope="module")
def support():
    model = load_g1_model()
    return model, FootSupport(model, G1Kinematics(model, dtype=torch.float64))


def measure(support, joints, weights=None, scalar=20.0):
    model, term = support
    target = torch.zeros(1, 1, 71, dtype=torch.float64)
    target[..., :36] = torch.tensor(model.qpos0)
    target[..., 2] = 1.0
    predicted = target.clone()
    for name, delta in joints:
        predicted[..., model.jnt_qposadr[model.joint(name).id]] += delta
    predicted.requires_grad_()
    loss, terms = term.loss(
        predicted,
        target,
        torch.ones(1, 1),
        0.0,
        0.0,
        effector_position_weight=scalar,
        effector_position_weights=weights,
    )
    gradient = torch.autograd.grad(loss, predicted)[0]
    return loss.detach(), terms.detach(), gradient


def test_ones_and_omitted_weights_have_exact_value_and_gradient_parity(support):
    a = measure(support, [("left_knee_joint", 0.2), ("left_elbow_joint", 0.25)])
    b = measure(support, [("left_knee_joint", 0.2), ("left_elbow_joint", 0.25)], (1.0,) * 4)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_compensated_foot_doubling_doubles_foot_only_error_and_gradient(support):
    old = measure(support, [("left_knee_joint", 0.2)])
    new = measure(support, [("left_knee_joint", 0.2)], (2.0, 2.0, 1.0, 1.0), 30.0)
    torch.testing.assert_close(new[0], 2 * old[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new[2], 2 * old[2], rtol=1e-12, atol=1e-12)


def test_compensated_foot_doubling_keeps_hand_only_error_and_gradient(support):
    old = measure(support, [("left_elbow_joint", 0.25)])
    new = measure(support, [("left_elbow_joint", 0.25)], (2.0, 2.0, 1.0, 1.0), 30.0)
    torch.testing.assert_close(new[0], old[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new[2], old[2], rtol=1e-12, atol=1e-12)


def test_only_relative_limb_weights_matter(support):
    a = measure(
        support, [("left_knee_joint", 0.2), ("left_elbow_joint", 0.25)], (2.0, 2.0, 1.0, 1.0)
    )
    b = measure(
        support, [("left_knee_joint", 0.2), ("left_elbow_joint", 0.25)], (20.0, 20.0, 10.0, 10.0)
    )
    torch.testing.assert_close(a[0], b[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(a[2], b[2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0, 1.0),
        (float("nan"), 1.0, 1.0, 1.0),
        (float("inf"), 1.0, 1.0, 1.0),
    ],
)
def test_invalid_weights_rejected_before_training_or_fk(support, tmp_path, weights):
    with pytest.raises(ValueError):
        measure(support, [], weights)
    with pytest.raises(ValueError):
        training_config("smoke", tmp_path, tmp_path / "out", effector_position_weights=weights)
    with pytest.raises(ValueError):
        PolicyTrainer(None, None, None, None, PolicyTrainerCfg(effector_position_weights=weights))


def recipe(tmp_path, weights=None):
    cfg = TrainConfig()
    saved = asdict(cfg)
    if weights is None:
        saved["policy_trainer"].pop("effector_position_weights", None)
    else:
        saved["policy_trainer"]["effector_position_weights"] = weights
    path = tmp_path / "recipe.pt"
    torch.save({"config": saved, "iteration": 1}, path)
    return path


def test_legacy_resume_keeps_equal_limb_default(tmp_path):
    cfg = training_config("smoke", tmp_path, tmp_path / "out", resume=recipe(tmp_path))
    assert cfg.policy_trainer.effector_position_weights == (1.0,) * 4


def test_resume_inherits_saved_limb_weights(tmp_path):
    cfg = training_config(
        "smoke", tmp_path, tmp_path / "out", resume=recipe(tmp_path, (2.0, 2.0, 1.0, 1.0))
    )
    assert cfg.policy_trainer.effector_position_weights == (2.0, 2.0, 1.0, 1.0)


def test_explicit_weights_override_resume_and_preset(tmp_path):
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "out",
        resume=recipe(tmp_path, (2.0, 2.0, 1.0, 1.0)),
        tracking_objective="contact",
        effector_position_weights=(3.0, 3.0, 1.0, 1.0),
    )
    assert cfg.policy_trainer.effector_position_weights == (3.0, 3.0, 1.0, 1.0)


def test_explicit_preset_restores_its_equal_limb_default(tmp_path):
    cfg = training_config(
        "smoke",
        tmp_path,
        tmp_path / "out",
        resume=recipe(tmp_path, (2.0, 2.0, 1.0, 1.0)),
        tracking_objective="contact",
    )
    assert cfg.policy_trainer.effector_position_weights == (1.0,) * 4


def test_cli_exposes_optional_ordered_positive_coefficients():
    args = parser().parse_args(["train", "--effector-position-weights", "2", "2", "1", "1"])
    assert args.effector_position_weights == [2.0, 2.0, 1.0, 1.0]
    assert parser().parse_args(["train"]).effector_position_weights is None
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--effector-position-weights", "2", "0", "1", "1"])


@pytest.mark.parametrize("legacy", [False, True])
def test_export_records_effective_limb_weights(tmp_path, legacy):
    from supermimic.deployment.export import export_policy

    cfg = TrainConfig()
    cfg.policy = PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,))
    cfg.policy_trainer.effector_position_weights = (2.0, 2.0, 1.0, 1.0)
    saved = asdict(cfg)
    if legacy:
        saved["policy_trainer"].pop("effector_position_weights")
    checkpoint = tmp_path / "actor.pt"
    torch.save(
        {"config": saved, "policy": Policy(cfg.policy).state_dict(), "iteration": 1}, checkpoint
    )
    export_policy(checkpoint, tmp_path / "export")
    contract = json.loads((tmp_path / "export/contract.json").read_text())
    assert contract["training_metadata"]["tracking_auxiliary"]["effector_position_weights"] == (
        [1.0] * 4 if legacy else [2.0, 2.0, 1.0, 1.0]
    )
