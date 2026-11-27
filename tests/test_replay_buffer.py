"""Episode boundaries and next-state labels must survive ring-buffer sampling."""

import torch

from supermimic.data.replay_buffer import ReplayBuffer


def fill_episode(buffer):
    for t in range(4):
        a = torch.full((1, 29), float(t))
        buffer.add(
            torch.tensor([0]),
            s=torch.full((1, 71), float(t)),
            a=a,
            a_exec=a,
            xi=torch.zeros(1, 33),
            r=torch.tensor([float(t)]),
            done=torch.tensor([float(t >= 2)]),
            frame=torch.tensor([t]),
            clip=torch.tensor([0]),
            terminal=torch.tensor([t == 3]),
            ep_start=torch.tensor([t == 0]),
        )


def test_terminal_transition_does_not_label_previous_state_early(monkeypatch):
    buffer = ReplayBuffer(1, 16, 71, 29, 33, "cpu")
    fill_episode(buffer)
    monkeypatch.setattr(
        buffer, "_sample_from_mask", lambda mask, n: (torch.tensor([0]), torch.tensor([0]))
    )
    short = buffer.sample_windows(1, 2)
    assert short.states[:, 0, 0].tolist() == [0, 1, 2]
    assert short.done[:, 0].tolist() == [0, 0]
    terminal = buffer.sample_windows(1, 3)
    assert terminal.states[:, 0, 0].tolist() == [0, 1, 2, 3]
    assert terminal.done[:, 0].tolist() == [0, 0, 1]


def test_sampled_windows_never_cross_episode_reset():
    buffer = ReplayBuffer(1, 16, 71, 29, 33, "cpu")
    fill_episode(buffer)
    buffer.end_episode(torch.tensor([0]))
    fill_episode(buffer)
    torch.manual_seed(7)
    windows = buffer.sample_windows(100, 3)
    assert torch.equal(windows.states[:, :, 0], torch.arange(4)[:, None].expand(4, 100))
    assert windows.done[:, 0].tolist() == [0, 0, 1]


def test_reset_history_backfills_state_and_zero_previous_action(monkeypatch):
    buffer = ReplayBuffer(1, 16, 71, 29, 33, "cpu")
    fill_episode(buffer)
    buffer.end_episode(torch.tensor([0]))
    fill_episode(buffer)
    monkeypatch.setattr(
        buffer, "_sample_from_mask", lambda mask, n: (torch.tensor([0]), torch.tensor([4]))
    )
    start = buffer.sample_starts(1, 10)
    assert not start.hist_states.any()
    assert not start.hist_actions.any()
    assert not start.prev_action.any()


def test_reset_curriculum_samples_real_reset_rows_with_correct_history():
    buffer = ReplayBuffer(1, 32, 71, 29, 33, "cpu")
    fill_episode(buffer)
    starts = buffer.sample_starts(20, 10, reset_start_fraction=1.0)
    assert not starts.state.any() and not starts.hist_states.any()
    assert not starts.hist_actions.any() and not starts.prev_action.any()
    mixed = buffer.sample_starts(20, 10, reset_start_fraction=0.5)
    assert (mixed.state[:, 0] == 0).sum() >= 10
    buffer.ep_start[:] = False
    assert buffer.sample_starts(20, 10, reset_start_fraction=1.0).state.shape == (20, 71)


def test_window_mask_keeps_ring_chronology_and_terminal_targets():
    buffer = ReplayBuffer(2, 11, 1, 1, 1, "cpu")
    buffer.valid[:] = True
    buffer.ptr[:] = torch.tensor([4, 8])
    buffer.ep_id[1, 8:] = 1
    buffer.ep_id[1, :2] = 1
    buffer.ep_id[1, 2:8] = 2
    buffer.terminal[1, 1] = True
    buffer.valid[1, 6] = False

    for length in (0, 1, 3, 4, 8, 10, 11):
        for env in range(buffer.E):
            expected = []
            ptr = int(buffer.ptr[env])
            for t in range(buffer.T):
                rows = [(t + step) % buffer.T for step in range(length + 1)]
                allowed = (
                    (ptr - t - 1) % buffer.T >= length
                    and all(buffer.valid[env, row] for row in rows)
                    and all(buffer.ep_id[env, row] == buffer.ep_id[env, t] for row in rows)
                    and not any(buffer.terminal[env, row] for row in rows[: max(length, 1)])
                )
                expected.append(allowed)
            assert buffer.window_mask(length)[env].tolist() == expected
