"""Motion consumers retain the validated NPZ publication when its path rotates."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.mark.parametrize("reader", ["reference", "library", "pack"])
def test_motion_consumers_read_validated_inode_after_path_replacement(
    tmp_path, monkeypatch, reader
):
    import numpy as np
    from test_motion_preparation import _motion

    from supermimic.data import schema

    source = tmp_path / "motions/clip.npz"
    replacement = tmp_path / "replacement.npz"
    _motion(source, frames=3)
    _motion(replacement, frames=7)
    original_open = schema.open_motion
    validated_info = []

    @contextmanager
    def replace_after_validation(path, **kwargs):
        # Perform real validation and retain the real NpzFile. Only the timing
        # of the concurrent publisher is controlled by this wrapper.
        with original_open(path, **kwargs) as (data, info):
            if Path(path) == source and not validated_info:
                replacement.replace(source)
                validated_info.append(info)
            yield data, info

    monkeypatch.setattr(schema, "open_motion", replace_after_validation)

    if reader == "reference":
        from supermimic.deployment import reference

        monkeypatch.setattr(reference, "open_motion", replace_after_validation)
        motion = reference.MotionReference(source, np.zeros(29, dtype=np.float32))
        assert motion.frames == motion.info["frames"] == 3
        assert motion.info["duration_s"] == 2 / 50
        assert motion.q.shape == (3, 29)
        assert motion.root_pos.shape == (3, 3)
        assert motion.features.shape == (3, 68)
        assert motion.body_names == motion.info["body_names"]
    elif reader == "library":
        from supermimic.data import loading
        from supermimic.data.motion_library import MotionLibrary

        monkeypatch.setattr(loading, "open_motion", replace_after_validation)
        library = MotionLibrary(
            [source], ("pelvis", "torso_link"), np.zeros(29), device="cpu", compact=False
        )
        assert library.num_frames == library.clips[0].length == 3
        assert library.num_clips == 1 and library.clips[0].name == "clip"
        assert library.joint_pos.shape == (3, 29)
        assert library.body_pos_w.shape == (3, 2, 3)
    else:
        import torch

        from supermimic import resources

        script = Path(__file__).resolve().parents[1] / "scripts/data/build_library.py"
        spec = importlib.util.spec_from_file_location("motion_reader_packer", script)
        packer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packer)
        # The fixture contains only a few KiB; isolate the packer's admission
        # estimate from unrelated concurrent jobs and small CI hosts.
        monkeypatch.setattr(
            resources, "host_snapshot", lambda: {"ram_available_bytes": 100 * resources.GIB}
        )
        output = tmp_path / "library.pt"
        packer._pack(source.parent, output, workers=1)
        packed = torch.load(output, map_location="cpu", weights_only=True)
        assert packed["names"] == ["clip"]
        assert packed["lengths"] == [3] and packed["starts"] == [0]
        assert packed["ref"].shape == (3, 71)
        assert packed["ref"].dtype == torch.float16

    assert validated_info and validated_info[0]["frames"] == 3
    assert validated_info[0]["duration_s"] == 2 / 50
    # The constructor/packer has returned and closed the old archive; the path
    # now refers to the new publication, while consumed arrays above stay old.
    with np.load(source, allow_pickle=False) as current:
        assert current["joint_pos"].shape == (7, 29)
