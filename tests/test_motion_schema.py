import numpy as np
import pytest
from test_deployment import make_motion

from supermimic.data.schema import validate_motion


def test_motion_schema_rejects_wrong_fps_joint_order_and_nan(tmp_path):
    path = tmp_path / "motion.npz"
    make_motion(path)
    assert validate_motion(path)["frames"] == 20
    with np.load(path) as values:
        valid = {key: values[key].copy() for key in values.files}
    for field, bad in [
        ("fps", [30.0]),
        ("joint_names", valid["joint_names"][::-1]),
        ("joint_pos", np.full((20, 29), np.nan)),
        ("body_quat_w", np.zeros((20, 30, 4))),
    ]:
        broken = tmp_path / (field + ".npz")
        np.savez(broken, **{**valid, field: bad})
        with pytest.raises(ValueError):
            validate_motion(broken)


def test_motion_unpack_budget_checked_before_arrays(tmp_path):
    path = tmp_path / "motion.npz"
    make_motion(path)
    with pytest.raises(ValueError, match="byte budget"):
        validate_motion(path, max_uncompressed_bytes=100)
