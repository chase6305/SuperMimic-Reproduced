"""Episode boundaries and next-state labels must survive ring-buffer sampling."""

import random

import pytest
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
        buffer, "_sample_indices", lambda candidates, n: (torch.tensor([0]), torch.tensor([0]))
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


def write_row(buffer, env, serial, episode, *, terminal=False, start=False):
    """Tiny state/action dimensions keep ring-boundary scenarios inexpensive."""
    ids = torch.tensor([env])
    buffer.add(
        ids,
        s=torch.tensor([[float(serial), float(episode)]]),
        a=torch.tensor([[serial + 10.0]]),
        a_exec=torch.tensor([[serial + 11.0]]),
        xi=torch.tensor([[float(episode)]]),
        r=torch.tensor([float(serial)]),
        done=torch.tensor([float(terminal)]),
        frame=torch.tensor([serial]),
        clip=torch.tensor([episode]),
        terminal=torch.tensor([terminal]),
        ep_start=torch.tensor([start]),
    )


def test_window_sampling_matches_chronological_oracle_through_overwrites():
    """Compare a ring against Python episode logs, including asynchronous resets."""
    rng = random.Random(19)
    buffer = ReplayBuffer(3, 13, 2, 1, 1, "cpu")
    logs = [[] for _ in range(3)]
    episodes, lengths, counts = [0] * 3, [0] * 3, [0] * 3
    for serial in range(100):
        env = rng.randrange(3)
        terminal = lengths[env] > 0 and rng.random() < 0.3
        row = (serial, episodes[env], terminal)
        write_row(buffer, env, serial, episodes[env], terminal=terminal, start=lengths[env] == 0)
        logs[env].append(row)
        logs[env] = logs[env][-buffer.T :]
        counts[env] += 1
        lengths[env] += 1
        if terminal:
            buffer.end_episode(torch.tensor([env]))
            episodes[env] += 1
            lengths[env] = 0
        for k in (1, 3, 8):
            expected = set()
            for e, rows in enumerate(logs):
                for i in range(len(rows) - k):
                    window = rows[i : i + k + 1]
                    if len({r[1] for r in window}) == 1 and not any(r[2] for r in window[:-1]):
                        expected.add((e, (counts[e] - len(rows) + i) % buffer.T))
            actual = {tuple(pair) for pair in buffer.window_mask(k).nonzero().tolist()}
            assert actual == expected
            assert buffer.has_windows(k) == bool(expected)
            if expected:
                sampled = buffer.sample_windows(12, k)
                for col in range(12):
                    assert sampled.states[:, col, 1].unique().numel() == 1
                    assert sampled.frame[:, col].tolist() == sampled.states[:-1, col, 0].tolist()
                    assert not sampled.done[:-1, col].any()
                    assert torch.equal(sampled.a_exec[:, col, 0], sampled.frame[:, col] + 11)
            else:
                with pytest.raises(RuntimeError, match="no sampleable rows"):
                    buffer.sample_windows(1, k)


def test_sample_history_matches_episode_log_after_multiple_ring_wraps():
    buffer = ReplayBuffer(1, 19, 2, 1, 1, "cpu")
    episode_rows = {}
    serial = 0
    for episode, length in enumerate((4, 24, 3, 15)):
        rows = []
        for t in range(length):
            write_row(buffer, 0, serial, episode, terminal=t == length - 1, start=t == 0)
            rows.append(serial)
            serial += 1
        episode_rows[episode] = rows
        buffer.end_episode(torch.tensor([0]))
    starts = buffer.sample_starts(256, 5, reset_start_fraction=0.5)
    for col in range(256):
        current, episode = (int(v) for v in starts.state[col])
        rows = episode_rows[episode]
        index = rows.index(current)
        expected_states = [rows[max(0, index - 4 + i)] for i in range(5)]
        expected_actions = [0 if s == rows[0] else s + 9 for s in expected_states]
        assert starts.hist_states[:, col, 0].tolist() == expected_states
        assert starts.hist_actions[:, col, 0].tolist() == expected_actions
        assert starts.prev_action[col, 0] == expected_actions[-1]


def test_window_index_cache_is_bounded_and_a_write_invalidates_empty_results():
    buffer = ReplayBuffer(1, 16, 2, 1, 1, "cpu")
    for k in range(1, 16):
        assert not buffer.has_windows(k)
    assert len(buffer._window_indices) <= 2
    write_row(buffer, 0, 0, 0, start=True)
    assert not buffer.has_windows(1)
    write_row(buffer, 0, 1, 0)
    assert buffer.has_windows(1)
    assert buffer.sample_windows(20, 1).states[:, :, 0].tolist() == [[0] * 20, [1] * 20]


def test_sampling_remains_uniform_over_rows_in_different_environments():
    buffer = ReplayBuffer(2, 16, 2, 1, 1, "cpu")
    for env, length in enumerate((2, 4)):
        for t in range(length):
            write_row(buffer, env, env * 10 + t, 0, start=t == 0)
    torch.manual_seed(32)
    frames = buffer.sample_windows(12000, 1).frame[0]
    counts = [(frames == value).sum().item() for value in (0, 10, 11, 12)]
    # Four eligible rows have equal probability; environments do not.
    assert all(2700 < count < 3300 for count in counts)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_sampling_rejects_invalid_counts_and_horizons(value):
    buffer = ReplayBuffer(1, 16, 2, 1, 1, "cpu")
    with pytest.raises(ValueError, match="Sample count"):
        buffer.sample_starts(value, 5)
    with pytest.raises(ValueError, match="History length"):
        buffer.sample_starts(1, value)
    with pytest.raises(ValueError, match="Replay window"):
        buffer.sample_windows(1, value)


def test_sampling_rejects_windows_or_histories_that_cannot_fit():
    buffer = ReplayBuffer(1, 16, 2, 1, 1, "cpu")
    for serial in range(buffer.T):
        write_row(buffer, 0, serial, 0, start=serial == 0)
    assert buffer.has_windows(15)
    assert not buffer.has_windows(16)
    assert not buffer.window_mask(16).any()
    with pytest.raises(RuntimeError, match="no sampleable rows"):
        buffer.sample_windows(1, 16)
    with pytest.raises(ValueError, match="below replay capacity"):
        buffer.sample_starts(1, 15)
