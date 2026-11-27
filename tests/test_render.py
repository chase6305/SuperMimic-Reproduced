"""Regression coverage for bounded CPU animation output."""

import sys
from contextlib import contextmanager

import pytest

from supermimic.deployment import render


def test_render_window_includes_one_state_and_clips_at_saved_end():
    assert list(render._frame_indices(1, 0.02, 0, 0.001, 25)) == [0]
    assert list(render._frame_indices(255, 0.02, 4.96, 5, 25)) == [248, 250, 252, 254]
    with pytest.raises(ValueError, match="outside"):
        render._frame_indices(255, 0.02, 1e308, 5, 25)


@pytest.mark.parametrize(("fps", "colors"), [(10, (20, 50, 80)), (25, (30, 90)), (50, (50,))])
def test_gif_preserves_frame_count_clock_and_loop(tmp_path, fps, colors):
    from PIL import Image

    source, target = tmp_path / "frames.mp4", tmp_path / "render.gif"
    with render._frame_writer(source, 64, 48, fps, lossless=True) as stream:
        for red in colors:
            stream.write(bytes((red, 100, 150)) * (64 * 48))
    render._encode_gif(source, target, fps)

    with Image.open(target) as animation:
        assert animation.size == (64, 48)
        assert animation.info["loop"] == 0
        assert animation.n_frames == len(colors)
        for index, red in enumerate(colors):
            animation.seek(index)
            assert animation.info["duration"] == 1000 // fps
            assert animation.convert("RGB").getpixel((0, 0)) == (red, 100, 150)


def test_frame_writer_reports_failed_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        render,
        "_ffmpeg_command",
        lambda: [sys.executable, "-c", "import sys; sys.exit('encoder diagnostic')"],
    )
    with (
        pytest.raises(RuntimeError, match="encoder diagnostic"),
        render._frame_writer(tmp_path / "bad.mp4", 64, 48, 25, lossless=False) as stream,
    ):
        stream.write(bytes(64 * 48 * 3))


def test_publish_preserves_existing_target_and_rolls_back_sidecars(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    poster, target = tmp_path / "preview.png", tmp_path / "preview.gif"
    new_poster, new_gif = stage / "poster.png", stage / "render.gif"
    new_poster.write_bytes(b"poster")
    new_gif.write_bytes(b"new GIF")
    target.write_bytes(b"existing GIF")
    with pytest.raises(FileExistsError):
        render._publish_render(((new_poster, poster), (new_gif, target)))
    assert not poster.exists()
    assert target.read_bytes() == b"existing GIF"


def test_render_destination_owns_concurrent_attempts_and_preserves_prior_evidence(tmp_path):
    target = tmp_path / "preview.gif"
    poster, metadata = tmp_path / "preview.gif.png", tmp_path / "preview.gif.json"
    resources = tmp_path / "preview.gif.resources.json"
    with render._reserve_render_output(target, poster, metadata):
        with (
            pytest.raises(FileExistsError, match="in progress"),
            render._reserve_render_output(target, poster, metadata),
        ):
            raise AssertionError("Two writers acquired one destination")
        target.write_bytes(b"existing GIF")
        resources.write_text('{"status":"complete"}')

    with pytest.raises(FileExistsError, match="preview.gif"):
        render.render_rollout(tmp_path / "bundle", tmp_path / "trajectory.npz", target)
    assert resources.read_text() == '{"status":"complete"}'
    assert sorted(tmp_path.iterdir()) == [target, resources]


def test_resource_stop_at_end_leaves_render_unpublished(tmp_path, monkeypatch):
    from supermimic import cpu_resources

    @contextmanager
    def stop_on_exit(*args, **kwargs):
        yield
        raise RuntimeError("resource guard stopped")

    def create_staged_animation(bundle, trajectory, target, poster, *args):
        target.write_bytes(b"animation")
        poster.write_bytes(b"poster")
        return {"status": "complete"}

    monkeypatch.setattr(cpu_resources, "cpu_guard", stop_on_exit)
    monkeypatch.setattr(render, "_render_animation", create_staged_animation)
    with pytest.raises(RuntimeError, match="resource guard stopped"):
        render.render_rollout(tmp_path / "bundle", tmp_path / "traj.npz", tmp_path / "out.gif")
    assert list(tmp_path.iterdir()) == []
