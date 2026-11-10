import numpy as np
import pytest

from supermimic.data.motion_convert import _so3_derivative, resample_motion


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_resampler_rejects_invalid_clock(fps):
    raw = np.zeros((2, 36))
    raw[:, 6] = 1
    with pytest.raises(ValueError, match="frame rates"):
        resample_motion(raw, fps, 50)


def test_resampler_rejects_zero_and_nan_quaternions():
    raw = np.zeros((2, 36))
    with pytest.raises(ValueError, match="quaternion"):
        resample_motion(raw, 30, 50)
    raw[:, 6] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_motion(raw, 30, 50)


def test_two_frame_rotation_has_nonzero_endpoint_velocity():
    theta, dt = 0.2, 0.02
    q = np.array([[1, 0, 0, 0], [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]])
    np.testing.assert_allclose(_so3_derivative(q, dt), [[0, 0, 10], [0, 0, 10]], atol=1e-12)


def test_resampler_checks_size_before_allocating():
    raw = np.zeros((2, 36))
    raw[:, 6] = 1
    with pytest.raises(ValueError, match="memory budget"):
        resample_motion(raw, 1e-10, 50)


def test_large_csv_rejected_before_parsing_or_allocating(tmp_path):
    from supermimic.data.motion_convert import MAX_INPUT_CSV_BYTES, load_csv

    path = tmp_path / "oversized.csv"
    with path.open("wb") as stream:
        stream.truncate(MAX_INPUT_CSV_BYTES + 1)
    with pytest.raises(ValueError, match="input memory budget"):
        load_csv(path)


def test_interrupted_conversion_keeps_previous_output(tmp_path, monkeypatch):
    from supermimic.data.motion_convert import convert_csv_to_npz

    raw = np.zeros((3, 36))
    raw[:, 2] = 0.78
    raw[:, 6] = 1
    source = tmp_path / "motion.csv"
    dest = tmp_path / "motion.npz"
    np.savetxt(source, raw, delimiter=",")
    dest.write_bytes(b"previous complete file")

    def failed_save(stream, **values):
        stream.write(b"partial new file")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr("supermimic.data.motion_convert.np.savez", failed_save)
    with pytest.raises(OSError, match="interrupted write"):
        convert_csv_to_npz(source, dest, input_fps=50)
    assert dest.read_bytes() == b"previous complete file"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["motion.csv", "motion.npz"]
