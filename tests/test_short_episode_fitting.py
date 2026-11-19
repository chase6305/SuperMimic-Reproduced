"""Supplementary one-step fitting must include failures absent from 8-step windows."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def test_world_model_learns_from_short_episodes_without_mixing_window_metrics():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    buffer = ReplayBuffer(1, 64, 71, 29, 33, "cpu")
    ids = torch.tensor([0])
    for length, marker in [(4, 100.0), (12, 0.0)]:
        for t in range(length):
            state = torch.zeros(1, 71)
            state[:, 0] = marker
            state[:, 2] = 0.78
            state[:, 3] = 1
            state[:, 7:36] = t * 0.001
            buffer.add(
                ids,
                state,
                torch.zeros(1, 29),
                torch.zeros(1, 29),
                torch.zeros(1, 33),
                torch.zeros(1),
                torch.tensor([float(t >= length - 2)]),
                torch.tensor([t]),
                torch.tensor([0]),
                torch.tensor([t == length - 1]),
                torch.tensor([t == 0]),
            )
        buffer.end_episode(ids)
    original = buffer.sample_windows
    counts = []

    def sample(n, k):
        w = original(n, k)
        counts.append((k, int((w.states[0, :, 0] > 50).sum())))
        return w

    buffer.sample_windows = sample

    def reference(frames):
        pos = torch.zeros(len(frames), 3)
        pos[:, 2] = 0.78
        quat = torch.zeros(len(frames), 4)
        quat[:, 0] = 1
        return SimpleNamespace(root_pos_w=pos, root_quat_w=quat)

    library = SimpleNamespace(ref=reference)
    heads = SimpleNamespace(termination_features=AnalyticHeads.termination_features)
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads, "cpu")
    trainer = WorldModelTrainer(
        wm,
        library,
        WMTrainerCfg(
            window=8,
            batch_size=128,
            batches_per_update=1,
            one_step_batches_per_update=2,
            compile=None,
        ),
    )
    metrics = trainer.update(buffer)
    assert counts[0] == (8, 0) and all(k == 1 and short > 0 for k, short in counts[1:])
    assert "one_step/nse_1step" in metrics and "nse_kstep" in metrics
    assert int(next(iter(trainer.opt.state.values()))["step"]) == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_metrics_average_each_window_group_without_changing_optimizer_steps():
    trainer = WorldModelTrainer.__new__(WorldModelTrainer)
    trainer.cfg = WMTrainerCfg(
        window=3, batches_per_update=2, one_step_batches_per_update=1, batch_size=2
    )
    trainer.wm = torch.nn.Linear(1, 1, bias=False)
    trainer.wm.weight.data.fill_(1)
    trainer.opt = torch.optim.Adam(trainer.wm.parameters(), lr=0)
    calls = []

    def loss_on(window):
        calls.append(window)
        loss = trainer.wm.weight.square().sum() + len(calls)
        return loss, {"probe": torch.tensor(float(10 * window + len(calls)))}

    trainer.loss_on = loss_on
    metrics = trainer.update(SimpleNamespace(sample_windows=lambda n, k: k))
    assert calls == [3, 3, 1]
    assert metrics == pytest.approx(
        {"loss": 2.5, "probe": 31.5, "grad_norm": 2,
         "one_step/loss": 4, "one_step/probe": 13, "one_step/grad_norm": 2}
    )
    assert int(trainer.opt.state[trainer.wm.weight]["step"]) == 3
    trainer.cfg.batches_per_update = trainer.cfg.one_step_batches_per_update = 0
    assert trainer.update(None) == {}
