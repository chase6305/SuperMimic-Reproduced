"""Assemble actor starts lazily from complete recorded episodes and warm histories.

Policy histories contain clean network outputs. Execution noise belongs to the
world-model transition data and is deliberately absent from this interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate
from numbers import Real

import torch

from supermimic.data.replay_buffer import StartBatch


@dataclass(frozen=True)
class RecordedEpisode:
    states: torch.Tensor  # (controls + 1, state_dim), reached terminal state included
    clean_actions: torch.Tensor  # (controls, action_dim), before execution noise
    frames: torch.Tensor  # (controls,), reference target used for each transition
    xi: torch.Tensor  # (context_dim,), constant physical context
    initial_states: torch.Tensor | None = None  # (history, state_dim), ending at states[0]
    initial_actions: torch.Tensor | None = None  # previous clean action at each history state


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class RecordedStartReplay:
    """Store each state/action once and gather only the requested batch history.

    A missing initial history denotes a physical episode reset: repeat the first
    state and use zero previous actions. A warm branch must provide both initial
    histories, including its first state and the clean action preceding it.
    Records must describe one uninterrupted episode with a continuous reference
    clock. File identity and physical evidence are admitted by the caller.

    The byte budget covers retained tensor storage, not transient input/conversion
    buffers or model allocations. Construction owns its packed storage; later
    mutations of input tensors cannot change sampled starts. Stored floating
    point values use float32, matching the actor's normal replay inputs.
    """

    def __init__(
        self,
        episodes: list[RecordedEpisode],
        history: int,
        *,
        device: str | torch.device | None = None,
        max_storage_bytes: int = 128 * 1024**2,
    ):
        _positive_integer(history, "History length")
        _positive_integer(max_storage_bytes, "Storage budget")
        if not episodes:
            raise ValueError("At least one recorded episode is required")
        dimensions = None
        counts, resets = [], []
        for episode in episodes:
            s, a, f, x = episode.states, episode.clean_actions, episode.frames, episode.xi
            if (
                s.ndim != 2
                or a.ndim != 2
                or x.ndim != 1
                or not s.is_floating_point()
                or not a.is_floating_point()
                or not x.is_floating_point()
                or not len(a)
                or s.shape[0] != len(a) + 1
                or min(s.shape[1], a.shape[1], len(x)) < 1
                or f.shape != (len(a),)
                or f.dtype != torch.long
            ):
                raise ValueError("Inconsistent recorded state/action/frame/context shapes or types")
            dims = (s.shape[1], a.shape[1], len(x))
            if dimensions is not None and dims != dimensions:
                raise ValueError("Recorded episode dimensions must agree")
            dimensions = dims
            hs, ha = episode.initial_states, episode.initial_actions
            reset = hs is None and ha is None
            if not reset:
                if (
                    hs is None
                    or ha is None
                    or hs.shape != (history, dims[0])
                    or ha.shape != (history, dims[1])
                ):
                    raise ValueError("Warm starts require both complete initial histories")
                if not hs.is_floating_point() or not ha.is_floating_point():
                    raise ValueError("Warm histories must contain floating point values")
            counts.append(len(a))
            resets.append(reset)
        state_dim, action_dim, context_dim = dimensions
        self.size = sum(counts)
        padded = self.size + len(episodes) * (history - 1)
        dense_context_bytes = 4 * self.size * context_dim
        shared_context_bytes = 4 * len(episodes) * context_dim + 8 * (len(episodes) - 1)
        self._shared_context = shared_context_bytes < dense_context_bytes
        estimated = 4 * padded * (state_dim + action_dim)
        estimated += min(dense_context_bytes, shared_context_bytes)
        estimated += 8 * (2 * self.size + sum(resets))
        if estimated > max_storage_bytes:
            raise ValueError(f"Recorded replay needs {estimated} bytes, exceeding storage budget")
        for episode in episodes:
            values = [episode.states, episode.clean_actions, episode.xi]
            if episode.initial_states is not None:
                values.extend([episode.initial_states, episode.initial_actions])
                if not torch.equal(episode.initial_states[-1], episode.states[0]):
                    raise ValueError("Warm history must end at the recorded first state")
            if not all(
                torch.isfinite(t).all() and (t.abs() <= torch.finfo(torch.float32).max).all()
                for t in values
            ):
                raise ValueError("Recorded values must be finite and representable as float32")
            f = episode.frames
            if (f < 0).any() or not torch.equal(f[1:], f[:-1] + 1):
                raise ValueError("Recorded reference frames must be nonnegative and continuous")
        self.device = torch.device(device or episodes[0].states.device)
        self.history = history
        states, actions, contexts, frames, rows, reset_rows = [], [], [], [], [], []
        offset = row_offset = 0

        def floating(tensor):
            return tensor.detach().to(device=self.device, dtype=torch.float32)

        for episode, count, reset in zip(episodes, counts, resets, strict=True):
            s, a, x = map(floating, (episode.states, episode.clean_actions, episode.xi))
            if reset:
                hs = s[:1].expand(history, -1)
                ha = a.new_zeros(history, action_dim)
                reset_rows.append(row_offset)
            else:
                hs, ha = floating(episode.initial_states), floating(episode.initial_actions)
            states.extend([hs[:-1], s[:-1]])
            actions.extend([ha[:-1], ha[-1:], a[:-1]])
            contexts.append(x[None] if self._shared_context else x.expand(count, -1))
            frames.append(episode.frames.detach().to(self.device))
            rows.append(torch.arange(count, device=self.device) + offset + history - 1)
            offset += count + history - 1
            row_offset += count
        self._states = torch.cat(states)
        self._actions = torch.cat(actions)
        self._xi = torch.cat(contexts)
        self._context_ends = torch.tensor(
            list(accumulate(counts))[:-1] if self._shared_context else [],
            device=self.device,
            dtype=torch.long,
        )
        self._frames = torch.cat(frames)
        self._rows = torch.cat(rows)
        self._reset_rows = torch.tensor(reset_rows, device=self.device, dtype=torch.long)
        self.storage_bytes = sum(
            t.numel() * t.element_size()
            for t in (
                self._states,
                self._actions,
                self._xi,
                self._context_ends,
                self._frames,
                self._rows,
                self._reset_rows,
            )
        )
        assert self.storage_bytes == estimated

    @torch.no_grad()
    def starts(self, indices: torch.Tensor, history: int) -> StartBatch:
        _positive_integer(history, "History length")
        if history > self.history:
            raise ValueError("Requested history exceeds the recorded initial history")
        if indices.dtype != torch.long or indices.ndim != 1 or not len(indices):
            raise ValueError("Start indices must be a nonempty vector of integers")
        indices = indices.to(self.device)
        if (indices < 0).any() or (indices >= self.size).any():
            raise ValueError("Start indices are outside recorded episodes")
        current = self._rows[indices]
        past = current[None, :] + torch.arange(1 - history, 1, device=self.device)[:, None]
        hs, ha = self._states[past], self._actions[past]
        context = (
            torch.bucketize(indices, self._context_ends, right=True)
            if self._shared_context
            else indices
        )
        return StartBatch(hs[-1], self._xi[context], self._frames[indices], ha[-1], hs, ha)

    @torch.no_grad()
    def sample_starts(self, n: int, hist: int, reset_start_fraction: float = 0.0) -> StartBatch:
        _positive_integer(n, "Sample count")
        _positive_integer(hist, "History length")
        if hist > self.history:
            raise ValueError("Requested history exceeds the recorded initial history")
        if (
            isinstance(reset_start_fraction, bool)
            or not isinstance(reset_start_fraction, Real)
            or not math.isfinite(reset_start_fraction)
            or not 0 <= reset_start_fraction <= 1
        ):
            raise ValueError("Reset start fraction must be finite and within [0, 1]")
        reset_count = int(n * reset_start_fraction) if len(self._reset_rows) else 0
        resets = (
            self._reset_rows[
                torch.randint(len(self._reset_rows), (reset_count,), device=self.device)
            ]
            if reset_count
            else self._reset_rows[:0]
        )
        rest = torch.randint(self.size, (n - reset_count,), device=self.device)
        return self.starts(torch.cat([resets, rest]), hist)
