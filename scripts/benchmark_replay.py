"""Bounded CPU replay timings, including cache rebuilds between fitting updates.

Run under the shared heavy-work lock. This measures replay sampling alone, not
training throughput or tracking quality. No simulator or motion data is needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, choices=(32, 128), default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30:
        parser.error("--repeats must be within [1, 30]")

    from supermimic.artifacts import write_json
    from supermimic.cpu_resources import cpu_guard
    from supermimic.resources import ResourceLimits

    with cpu_guard(args.output, limits=ResourceLimits(max_rss_gib=2), max_seconds=120):
        import torch

        from supermimic.data.replay_buffer import ReplayBuffer

        torch.set_num_threads(1)
        torch.manual_seed(42)
        buffer = ReplayBuffer(args.envs, 1000, 71, 29, 33, "cpu")
        ids = torch.arange(args.envs)
        # Populate through the write API, then continue writing between updates.
        state = torch.zeros(args.envs, 71)
        action = torch.zeros(args.envs, 29)
        context = torch.zeros(args.envs, 33)
        scalar = torch.zeros(args.envs)
        frame = torch.zeros(args.envs, dtype=torch.long)
        flag = torch.zeros(args.envs, dtype=torch.bool)

        def collect():
            buffer.add(ids, state, action, action, context, scalar, scalar, frame, frame, flag, flag)

        for _ in range(buffer.T):
            collect()

        def fit_update():
            collect()
            for _ in range(10):
                buffer.sample_windows(512, 8)
            for _ in range(2):
                buffer.sample_windows(512, 1)

        timings = {}
        for name, operation in (
            ("window_8_warm", lambda: buffer.sample_windows(512, 8)),
            ("window_1_warm", lambda: buffer.sample_windows(512, 1)),
            ("starts", lambda: buffer.sample_starts(48, 10)),
            ("collect_plus_12_minibatches", fit_update),
        ):
            for _ in range(3):
                operation()
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                for _ in range(10):
                    operation()
                samples.append((time.perf_counter() - started) * 1000 / 10)
            timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
        result = {
            "device": "cpu",
            "torch_version": torch.__version__,
            "threads": torch.get_num_threads(),
            "envs": args.envs,
            "capacity": buffer.T,
            "window_batch_size": 512,
            "warmup_operations": 3,
            "operations_per_sample": 10,
            "timings": timings,
            "scope": "sampling only; update includes one ring write, ten 8-step and two 1-step batches",
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
