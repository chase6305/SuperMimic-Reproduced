"""Trainer owns resources through constructor, signal setup and run failures."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from supermimic.config import TrainConfig
from supermimic.training import loop


@pytest.fixture
def resources(tmp_path, monkeypatch):
    events = []
    cfg = TrainConfig(log_root=tmp_path, device="cpu")

    class Writer:
        def __init__(self, *args):
            events.append("writer_open")

        def close(self):
            events.append("writer_close")

    monkeypatch.setattr(loop, "SummaryWriter", Writer)
    monkeypatch.setattr(loop, "init_distributed", lambda: (0, 0, 1))
    monkeypatch.setattr(loop, "finalize", lambda: events.append("distributed_close"))
    return cfg, events


def test_missing_motion_closes_writer_and_distributed_state(resources, monkeypatch):
    cfg, events = resources
    monkeypatch.setattr(loop, "discover_npz", lambda *args, **kwargs: [])
    with pytest.raises(FileNotFoundError, match="no clips"):
        loop.Trainer(cfg)
    assert events == ["writer_open", "writer_close", "distributed_close"]


@pytest.mark.parametrize("close_error", [False, True])
def test_post_environment_failure_closes_all_owned_resources(
    resources, tmp_path, monkeypatch, close_error
):
    cfg, events = resources
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"selected motion fingerprint")
    monkeypatch.setattr(loop, "discover_npz", lambda *args, **kwargs: [motion])

    def missing_command(name):
        raise RuntimeError("command initialization failed")

    def close_env():
        events.append("environment_close")
        if close_error:
            raise RuntimeError("environment cleanup failed")

    env = SimpleNamespace(
        close=close_env, command_manager=SimpleNamespace(get_term=missing_command)
    )
    monkeypatch.setattr(loop, "make_env", lambda *args, **kwargs: env)
    with pytest.raises(
        RuntimeError, match="cleanup failed" if close_error else "initialization failed"
    ):
        loop.Trainer(cfg)
    assert events == ["writer_open", "environment_close", "writer_close", "distributed_close"]


@pytest.fixture
def minimal_trainer():
    def create(*, run_error=False, close_error=False):
        events = []

        class MinimalTrainer(loop.Trainer):
            def _initialize(self, cfg, resources):
                self.stop_reason = None
                resources.callback(events.append, "distributed_close")
                resources.callback(close_writer)
                resources.callback(events.append, "environment_close")

            def _run(self):
                events.append("run")
                if run_error:
                    raise RuntimeError("training failed")

        def close_writer():
            events.append("writer_close")
            if close_error:
                raise RuntimeError("writer cleanup failed")

        return MinimalTrainer(None), events

    return create


@pytest.fixture
def signal_handlers(monkeypatch):
    original = {loop.signal.SIGINT: object(), loop.signal.SIGTERM: object()}
    installed = dict(original)

    def set_handler(sig, handler):
        previous = installed[sig]
        installed[sig] = handler
        return previous

    monkeypatch.setattr(loop.signal, "signal", set_handler)
    return original, installed


@pytest.mark.parametrize("run_error", [False, True])
def test_run_releases_transferred_resources_and_restores_handlers(
    minimal_trainer, signal_handlers, run_error
):
    trainer, events = minimal_trainer(run_error=run_error)
    assert events == []  # Construction transfers ownership without closing resources.

    if run_error:
        with pytest.raises(RuntimeError, match="training failed"):
            trainer.run()
    else:
        trainer.run()

    assert events == ["run", "environment_close", "writer_close", "distributed_close"]
    original, installed = signal_handlers
    assert installed == original
    trainer.close()
    assert events == ["run", "environment_close", "writer_close", "distributed_close"]


def test_second_signal_install_failure_restores_first_and_closes_resources(
    minimal_trainer, signal_handlers, monkeypatch
):
    trainer, events = minimal_trainer()
    original, installed = signal_handlers
    set_handler = loop.signal.signal

    def fail_second_install(sig, handler):
        if sig == loop.signal.SIGTERM:
            raise ValueError("signal installation failed")
        return set_handler(sig, handler)

    monkeypatch.setattr(loop.signal, "signal", fail_second_install)
    with pytest.raises(ValueError, match="signal installation failed"):
        trainer.run()

    assert installed == original
    assert events == ["environment_close", "writer_close", "distributed_close"]


@pytest.mark.parametrize("close_error", [False, True])
def test_close_is_idempotent_even_after_callback_failure(minimal_trainer, close_error):
    trainer, events = minimal_trainer(close_error=close_error)
    if close_error:
        with pytest.raises(RuntimeError, match="writer cleanup failed"):
            trainer.close()
    else:
        trainer.close()

    assert events == ["environment_close", "writer_close", "distributed_close"]
    trainer.close()
    assert events == ["environment_close", "writer_close", "distributed_close"]


@pytest.mark.parametrize("close_error", [False, True])
def test_closed_trainer_rejects_run_before_signals_or_training(
    minimal_trainer, monkeypatch, close_error
):
    trainer, events = minimal_trainer(close_error=close_error)
    if close_error:
        with pytest.raises(RuntimeError, match="writer cleanup failed"):
            trainer.close()
    else:
        trainer.close()
    closed_events = list(events)
    signal_change = Mock(side_effect=AssertionError("closed Trainer changed signal handlers"))
    monkeypatch.setattr(loop.signal, "signal", signal_change)

    with pytest.raises(RuntimeError, match="closed Trainer"):
        trainer.run()

    signal_change.assert_not_called()
    assert events == closed_events
