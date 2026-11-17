"""Recorded histories agree with streaming deployment and respect episode boundaries."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from supermimic.data.recorded_replay import RecordedEpisode, RecordedStartReplay
from supermimic.deployment.core import ObservationHistory
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


def episode(controls=5, offset=0):
    states = torch.zeros(controls + 1, 71)
    states[:, 2] = 0.75
    states[:, 3] = 1
    states[:, 7:36] = offset + torch.arange(controls + 1)[:, None] * 0.1
    states[:, 39:42] = torch.arange(controls + 1)[:, None] * 0.02
    states[:, 42:] = torch.arange(controls + 1)[:, None] * 0.03
    actions = torch.arange(controls * 29).reshape(controls, 29).float() * 0.01
    xi = torch.zeros(33)
    xi[0] = 1
    return RecordedEpisode(states, actions, torch.arange(1, controls + 1), xi)


def append(history, state, previous):
    observation = np.r_[state[7:36], state[42:], state[39:42], [0, 0, -1], previous]
    return history.append(observation)


def policy_history(batch):
    observations = analytic_observation(
        batch.hist_states, batch.hist_actions, batch.xi, torch.zeros(29)
    )
    return flatten_history_term_major(observations)


def test_reset_history_matches_streaming_deployment_with_clean_previous_actions():
    data = episode()
    replay = RecordedStartReplay([data], 4)
    batch = replay.starts(torch.arange(5), 4)
    history = ObservationHistory(4)
    expected = []
    for t in range(5):
        previous = data.clean_actions[t - 1].numpy() if t else np.zeros(29)
        expected.append(append(history, data.states[t].numpy(), previous))
    torch.testing.assert_close(
        policy_history(batch), torch.from_numpy(np.stack(expected)), atol=0, rtol=0
    )
    assert torch.equal(batch.prev_action[0], torch.zeros(29))
    assert torch.equal(batch.prev_action[1:], data.clean_actions[:-1])
    assert torch.equal(batch.frame, data.frames)


def test_warm_branch_preserves_real_prefix_then_uses_its_own_clean_actions():
    parent = episode(9)
    branch_states = parent.states[5:9].clone()
    branch_states[1:, 7:36] += 0.007
    clean = torch.tensor([2.0, 3.0, 4.0])[:, None].expand(-1, 29).clone()
    branch = RecordedEpisode(
        branch_states,
        clean,
        torch.arange(6, 9),
        parent.xi,
        parent.states[2:6],
        parent.clean_actions[1:5],
    )
    replay = RecordedStartReplay([branch], 4)
    batch = replay.starts(torch.arange(3), 4)
    history = ObservationHistory(4)
    for t in range(5):
        append(
            history,
            parent.states[t].numpy(),
            parent.clean_actions[t - 1].numpy() if t else np.zeros(29),
        )
    expected = []
    for t in range(3):
        previous = clean[t - 1].numpy() if t else parent.clean_actions[4].numpy()
        expected.append(append(history, branch_states[t].numpy(), previous))
    torch.testing.assert_close(
        policy_history(batch), torch.from_numpy(np.stack(expected)), atol=0, rtol=0
    )
    shorter = replay.starts(torch.arange(3), 2)
    assert torch.equal(shorter.hist_states, batch.hist_states[-2:])
    assert torch.equal(shorter.hist_actions, batch.hist_actions[-2:])
    # An all-warm archive follows the normal replay's fallback to uniform starts.
    sampled = replay.sample_starts(8, 4, 1)
    assert sampled.frame.min() >= 6 and sampled.frame.max() <= 8


def test_separate_episode_reset_does_not_inherit_the_previous_episode_tail():
    first, second = episode(3), episode(2, offset=100)
    replay = RecordedStartReplay([first, second], 5)
    batch = replay.starts(torch.tensor([3, 1, 4, 0]), 5)
    assert torch.equal(batch.hist_states[:, 0], second.states[:1].expand(5, -1))
    assert torch.count_nonzero(batch.hist_actions[:, 0]) == 0
    assert torch.equal(batch.hist_states[-1, 2], second.states[1])
    assert torch.equal(batch.prev_action[2], second.clean_actions[0])
    only_resets = replay.sample_starts(20, 5, 1)
    assert torch.count_nonzero(only_resets.prev_action) == 0
    assert set(only_resets.state[:, 7].tolist()) <= {0.0, 100.0}


def test_packed_storage_is_owned_and_avoids_materialized_history_copies():
    data = episode(200)
    replay = RecordedStartReplay([data], 10)
    before = replay.starts(torch.tensor([0, 99]), 10)
    dense_history_bytes = 200 * 10 * (71 + 29) * 4
    assert replay.storage_bytes < dense_history_bytes // 4
    data.states.fill_(900)
    data.clean_actions.fill_(800)
    data.frames.add_(100)
    data.xi.fill_(700)
    after = replay.starts(torch.tensor([0, 99]), 10)
    for name in ["state", "xi", "frame", "prev_action", "hist_states", "hist_actions"]:
        assert torch.equal(getattr(before, name), getattr(after, name))


def test_storage_budget_is_checked_before_value_scans_or_device_allocations(monkeypatch):
    data = episode()
    monkeypatch.setattr(torch, "isfinite", lambda *_: pytest.fail("scanned over-budget inputs"))
    with pytest.raises(ValueError, match="exceeding storage budget"):
        RecordedStartReplay([data], 10, device="cuda", max_storage_bytes=1)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_prefix_actions",
        "wrong_prefix_state",
        "frame_gap",
        "nonfinite_action",
        "float32_overflow",
    ],
)
def test_invalid_physical_history_cannot_silently_be_sampled(problem):
    data = episode()
    if problem == "missing_prefix_actions":
        data = replace(data, initial_states=data.states[:1].expand(4, -1))
    elif problem == "wrong_prefix_state":
        data = replace(data, initial_states=torch.ones(4, 71), initial_actions=torch.zeros(4, 29))
    elif problem == "frame_gap":
        data.frames[2] += 1
    elif problem == "nonfinite_action":
        data.clean_actions[0, 0] = float("nan")
    else:
        data = replace(data, states=data.states.double())
        data.states[0, 0] = 1e100
    with pytest.raises(ValueError):
        RecordedStartReplay([data], 4)


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([-1]),
        torch.tensor([5]),
        torch.tensor([1.0]),
        torch.empty(0, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.long),
    ],
)
def test_out_of_range_or_ambiguous_start_indices_are_rejected(indices):
    replay = RecordedStartReplay([episode()], 4)
    with pytest.raises(ValueError):
        replay.starts(indices, 4)


def test_invalid_sampling_options_do_not_consume_the_rng():
    replay = RecordedStartReplay([episode()], 4)
    rng = torch.get_rng_state().clone()
    for count, history, fraction in [
        (0, 4, 0),
        (2, 5, 0),
        (2, 0, 0),
        (2, 4, float("nan")),
        (2, 4, 1.1),
    ]:
        with pytest.raises(ValueError):
            replay.sample_starts(count, history, fraction)
        assert torch.equal(torch.get_rng_state(), rng)
