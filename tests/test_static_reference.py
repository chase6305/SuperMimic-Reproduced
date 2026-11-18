"""Fixed-pose references retain exact source poses and clocks, with zero velocities."""

import numpy as np
import pytest
from test_deployment import make_motion

from supermimic.artifacts import sha256
from supermimic.data.schema import open_motion
from supermimic.data.static_reference import make_static_reference
from supermimic.eval.static_hold import validate_static_reference


def test_selected_frame_is_frozen_at_original_fifty_hz_with_source_identity(tmp_path):
    source, output = tmp_path / "motion.npz", tmp_path / "static.npz"
    make_motion(source)
    digest = sha256(source)
    result = make_static_reference(source, output, frame=7, seconds=30)
    assert result["frames"] == 1501 and result["duration_s"] == 30 and result["fps"] == 50
    assert result["source_motion_sha256"] == digest and result["source_frame"] == 7
    with open_motion(output) as (actual, _), np.load(source) as expected:
        validate_static_reference(actual)
        assert actual["source_motion_sha256"].item() == digest
        assert actual["source_frame"].item() == 7
        for name in ("joint_pos", "body_pos_w", "body_quat_w"):
            value = actual[name]
            assert value.dtype == expected[name].dtype
            np.testing.assert_array_equal(value[0], expected[name][7])
            np.testing.assert_array_equal(value[-1], expected[name][7])
        for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            assert not actual[name].any()
        for name in ("fps", "joint_names", "body_names"):
            np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("seconds", [True, float("nan"), float("inf"), 29.99, 1800.1, 30.001])
def test_invalid_duration_rejected_without_opening_source(tmp_path, seconds):
    with pytest.raises(ValueError, match="seconds|duration"):
        make_static_reference(
            tmp_path / "absent.npz", tmp_path / "out.npz", frame=0, seconds=seconds
        )
    assert not (tmp_path / "out.npz").exists()


@pytest.mark.parametrize("frame", [True, -1, 1.5])
def test_invalid_frame_type_rejected_before_source_read(tmp_path, frame):
    with pytest.raises(ValueError, match="frame"):
        make_static_reference(
            tmp_path / "absent.npz", tmp_path / "out.npz", frame=frame, seconds=30
        )


def test_out_of_range_frame_and_existing_output_are_never_published_over(tmp_path):
    source, output = tmp_path / "motion.npz", tmp_path / "out.npz"
    make_motion(source)
    with pytest.raises(ValueError, match="outside"):
        make_static_reference(source, output, frame=20, seconds=30)
    assert not output.exists()
    assert not list(tmp_path.glob(".supermimic-static-*"))
    output.write_bytes(b"existing user artifact")
    with pytest.raises(FileExistsError):
        make_static_reference(source, output, frame=0, seconds=30)
    assert output.read_bytes() == b"existing user artifact"


def test_racing_output_publication_is_preserved(tmp_path, monkeypatch):
    import supermimic.data.static_reference as builder

    source, output = tmp_path / "motion.npz", tmp_path / "out.npz"
    make_motion(source)
    link = builder.os.link

    def publish_first(temporary, destination):
        destination.write_bytes(b"another writer won")
        link(temporary, destination)

    monkeypatch.setattr(builder.os, "link", publish_first)
    with pytest.raises(FileExistsError):
        make_static_reference(source, output, frame=0, seconds=30)
    assert output.read_bytes() == b"another writer won"
    assert not list(tmp_path.glob(".supermimic-static-*"))
