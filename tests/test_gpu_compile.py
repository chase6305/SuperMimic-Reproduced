"""Opt-in compiler numerical/performance probe; always run through the resource guard."""

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SUPERMIMIC_COMPILE_TESTS") != "1" or not torch.cuda.is_available(),
    reason="opt-in bounded CUDA compilation probe",
)


def test_compiled_fk_matches_outputs_and_gradients():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics

    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = load_g1_model()
    fk = G1Kinematics(model, "cuda:0")
    q = torch.tensor(model.qpos0, device="cuda:0", dtype=torch.float32).repeat(16, 1)
    q[:, 7:] += 0.05 * torch.randn_like(q[:, 7:])
    q.requires_grad_()
    v = (torch.randn(16, 35, device="cuda:0") * 0.1).requires_grad_()
    compiled = torch.compile(fk.forward, dynamic=False, fullgraph=True)
    expected = fk(q, v)
    actual = compiled(q, v)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
    ga = torch.autograd.grad(sum(value.square().mean() for value in actual.values()), (q, v))
    ge = torch.autograd.grad(sum(value.square().mean() for value in expected.values()), (q, v))
    for actual_grad, expected_grad in zip(ga, ge, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=1e-4)
    timings = {}
    for name, function in [("eager", fk.forward), ("compiled", compiled)]:
        with torch.no_grad():
            for _ in range(3):
                function(q, v)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(30):
                function(q, v)
            torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - started) / 30
    print("bounded FK timings (seconds/batch)", timings)


def test_compiled_world_model_window_loss_and_parameter_gradients():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(29)
    model = load_g1_model()
    heads = AnalyticHeads(
        G1Kinematics(model, "cuda:0"),
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        torch.tensor(model.jnt_range[1:], device="cuda:0", dtype=torch.float32),
        torch.tensor(model.qpos0[7:], device="cuda:0", dtype=torch.float32),
    )
    wm = WorldModel(WorldModelCfg(hidden=(64, 64)), heads, "cuda:0")
    with torch.no_grad():
        # Exercise all dynamics layers, rather than its zero-output initialization.
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.01)
        wm.dynamics.mlp.net[-1].bias.normal_(std=0.001)
    states = torch.zeros(9, 16, 71, device="cuda:0")
    states[..., :36] = torch.tensor(model.qpos0, device="cuda:0", dtype=torch.float32)
    states[..., :3] += 0.01 * torch.randn_like(states[..., :3])
    states[..., 7:] += 0.02 * torch.randn_like(states[..., 7:])
    actions = 0.1 * torch.randn(8, 16, 29, device="cuda:0")
    xi = torch.zeros(16, 33, device="cuda:0")
    xi[:, 0] = 0.75
    done = (torch.rand(8, 16, device="cuda:0") < 0.1).float()
    refs = [ReferenceFrame(None, None, None, None, s[..., :3], s[..., 3:7]) for s in states[1:]]
    parameters = tuple(wm.parameters())
    wm.enable_compile(None)
    expected, _ = wm.window_loss(states, actions, xi, done, refs)
    expected_gradients = torch.autograd.grad(expected, parameters)
    assert expected_gradients[0].abs().sum() > 0
    wm.enable_compile("default")
    actual, _ = wm.window_loss(states, actions, xi, done, refs)
    actual_gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, atol=3e-5, rtol=2e-4)
