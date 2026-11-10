"""Optional XY gradients, masking, old-checkpoint compatibility and actor wiring."""

import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.cli import parser
from supermimic.config import TrainConfig
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import ImagRollout
from supermimic.imagination.tracking import tracking_auxiliary
from supermimic.memory import estimate_training_memory
from supermimic.profiles import training_config


def test_xy_closed_form_gradient_detached_survival_and_dead_infinity():
    p = torch.zeros(1, 3, 71, dtype=torch.float64)
    p[..., 3] = 1
    p[0, 0, :2] = torch.tensor([3.0, 4.0])
    p[0, 1, :2] = torch.tensor([1.0, 2.0])
    t = torch.zeros_like(p)
    t[..., 3] = 1
    p[0, 2] = torch.inf
    p.requires_grad_()
    s = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64, requires_grad=True)
    loss, terms = tracking_auxiliary(p, t, s, (0.0, 0.0, 0.0, 0.0), dt=0.02, root_xy_weight=2.0)
    assert terms.shape == (5,)
    assert float(loss.detach()) == pytest.approx(0.02 * 2 * (25 + 0.5 * 5) / 1.5)
    loss.backward()
    torch.testing.assert_close(
        p.grad[0, 0, :2], torch.tensor([0.16, 0.2133333333333333], dtype=p.dtype)
    )
    torch.testing.assert_close(
        p.grad[0, 1, :2], torch.tensor([0.02666666666666667, 0.05333333333333334], dtype=p.dtype)
    )
    assert torch.isfinite(p.grad).all() and p.grad[0, 2].abs().sum() == 0
    assert p.grad[..., 2:].abs().sum() == 0 and s.grad is None


def test_xy_common_translation_and_yaw_do_not_change_loss():
    p = torch.zeros(2, 3, 71, dtype=torch.float64)
    p[..., 3] = 1
    t = p.clone()
    p[..., :2] = torch.tensor([3.0, 4.0])
    t[..., :2] = torch.tensor([1.0, -2.0])
    s = torch.ones(2, 3, dtype=p.dtype)
    before, _ = tracking_auxiliary(p, t, s, (0.0, 0.0, 0.0, 0.0), root_xy_weight=1.0)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=p.dtype)
    for v in (p, t):
        v[..., :2] = v[..., :2] @ rotation.T + torch.tensor([37.0, -11.0])
    after, _ = tracking_auxiliary(p, t, s, (0.0, 0.0, 0.0, 0.0), root_xy_weight=1.0)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_zero_xy_has_exact_original_loss_terms_and_gradients():
    spec = importlib.util.spec_from_file_location(
        "original_tracking", Path(__file__).parent / "fixtures/legacy_tracking_auxiliary.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(71)
    p = torch.randn(2, 3, 71, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(p)
    s = torch.tensor([[1.0, 0.5, 0.0], [0.7, 1.0, 0.2]], dtype=p.dtype)
    kwargs = dict(orientation_mode="geodesic", angular_velocity_weight=0.1)
    a, at = original.tracking_auxiliary(p, t, s, (3.0, 2.0, 1.0, 0.5), **kwargs)
    ag = torch.autograd.grad(a, p)[0]
    b, bt = tracking_auxiliary(p, t, s, (3.0, 2.0, 1.0, 0.5), root_xy_weight=0.0, **kwargs)
    bg = torch.autograd.grad(b, p)[0]
    assert torch.equal(a, b) and torch.equal(at, bt) and torch.equal(ag, bg)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_xy_rejected_by_profile_trainer_and_cli(tmp_path, bad):
    with pytest.raises(ValueError, match="Root XY"):
        training_config("smoke", tmp_path, tmp_path / "out", root_xy_tracking_weight=bad)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PolicyTrainer(None, None, None, None, PolicyTrainerCfg(root_xy_tracking_weight=bad))
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--root-xy-tracking-weight=" + str(bad)])


def test_xy_old_resume_inheritance_override_and_default_off(tmp_path):
    cfg = TrainConfig()
    assert cfg.policy_trainer.root_xy_tracking_weight == 0
    saved = asdict(cfg)
    saved["policy_trainer"].pop("root_xy_tracking_weight")
    old = tmp_path / "old.pt"
    torch.save({"config": saved, "iteration": 1}, old)
    assert (
        training_config(
            "smoke", tmp_path, tmp_path / "a", resume=old
        ).policy_trainer.root_xy_tracking_weight
        == 0
    )
    saved["policy_trainer"]["root_xy_tracking_weight"] = 1.5
    new = tmp_path / "new.pt"
    torch.save({"config": saved, "iteration": 1}, new)
    inherited = training_config("smoke", tmp_path, tmp_path / "b", resume=new)
    assert inherited.policy_trainer.root_xy_tracking_weight == 1.5
    override = training_config(
        "smoke", tmp_path, tmp_path / "c", resume=new, root_xy_tracking_weight=0
    )
    assert override.policy_trainer.root_xy_tracking_weight == 0
    baseline = training_config(
        "smoke", tmp_path, tmp_path / "d", resume=new, tracking_objective="baseline"
    )
    assert baseline.policy_trainer.root_xy_tracking_weight == 0
    enabled = training_config("smoke", tmp_path, tmp_path / "e", root_xy_tracking_weight=1)
    disabled = training_config("smoke", tmp_path, tmp_path / "f")
    assert enabled.policy == disabled.policy and enabled.world_model == disabled.world_model
    assert enabled.env == disabled.env
    assert (
        estimate_training_memory(enabled, 100)["components_mib"]["root_xy_tracking_workspace"] >= 1
    )
    assert (
        estimate_training_memory(disabled, 100)["components_mib"]["root_xy_tracking_workspace"] == 0
    )


def test_xy_only_actor_penalty_reaches_policy_gradient_and_metric(monkeypatch):
    import supermimic.imagination.policy_trainer as module

    model = PolicyTrainer.__new__(PolicyTrainer)
    model.cfg = PolicyTrainerCfg(
        horizon=2, batch_size=1, root_xy_tracking_weight=1.0, use_critic=False, obs_noise=False
    )
    model.policy = torch.nn.Linear(1, 1, bias=False)
    model.policy.weight.data.fill_(0.2)
    model.wm = SimpleNamespace(
        spec=SimpleNamespace(normalize_xi=lambda x: x, canonical=lambda x: x, dt=0.02)
    )
    model.step_fn = model.foot_support = None

    def reference(frames):
        out = torch.zeros(len(frames), 71)
        out[:, 3] = 1
        return out

    model.lib = SimpleNamespace(ref71=reference)

    def imagine(wm, policy, lib, start, horizon, future_frames, **kwargs):
        action = policy(torch.ones(1, 1)).reshape(1)
        base = torch.zeros(horizon + 1, 1, 71)
        base[..., 3] = 1
        direction = torch.zeros(71)
        direction[0] = 1
        return ImagRollout(
            states=base + action * direction,
            actions=action.expand(horizon, 1, 29),
            rewards=torch.zeros(horizon, 1),
            dhat=torch.zeros(horizon + 1, 1),
            survival=torch.ones(horizon + 1, 1),
            goals=torch.zeros(horizon + 1, 1, 2),
            e_root=torch.zeros(horizon + 1, 1, 4),
            frames=torch.zeros(horizon + 1, 1, dtype=torch.long),
            reward_terms={},
        )

    monkeypatch.setattr(module, "imagine", imagine)
    buffer = SimpleNamespace(sample_starts=lambda *args: SimpleNamespace(xi=torch.zeros(1, 33)))
    metrics, batch = model._actor_microbatch(buffer)
    assert batch is None and metrics["tracking_aux/root_xy_mse"] == pytest.approx(0.04)
    torch.testing.assert_close(model.policy.weight.grad, torch.tensor([[0.008]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bounded CUDA tracking-gradient check")
def test_compiled_xy_tracking_matches_cuda_outputs_and_gradients():
    torch.manual_seed(81)
    p = torch.randn(2, 4, 71, device="cuda:0", requires_grad=True)
    t = torch.randn_like(p)
    survival = torch.tensor([[1.0, 0.7, 0.4, 0.0], [0.8, 1.0, 0.0, 0.2]], device="cuda:0")

    def objective(prediction):
        return tracking_auxiliary(
            prediction,
            t,
            survival,
            (3.0, 2.0, 1.0, 0.5),
            orientation_mode="geodesic",
            angular_velocity_weight=0.1,
            root_xy_weight=1.0,
        )

    expected, expected_terms = objective(p)
    expected_gradient = torch.autograd.grad(expected, p)[0]
    compiled = torch.compile(objective, fullgraph=True, dynamic=False)
    actual, actual_terms = compiled(p)
    actual_gradient = torch.autograd.grad(actual, p)[0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_terms, expected_terms, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-5, rtol=2e-4)
    assert torch.isfinite(actual_gradient).all()
