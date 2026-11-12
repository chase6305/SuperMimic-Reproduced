"""Reference records and gathering do not depend on a world-model trainer."""

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from supermimic.world_model.reference import ReferenceFrame, reference_at


def _isolated_imports(code: str, blocked: list[str]) -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    guard = """
import importlib.abc
import json
import sys
sys.path.insert(0, sys.argv[1])
blocked = set(json.loads(sys.argv[2]))
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected reference dependency: ' + fullname)
sys.meta_path.insert(0, BlockImports())
"""
    subprocess.run(
        [sys.executable, "-I", "-c", guard + code, str(source), json.dumps(blocked)],
        check=True,
        timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_reference_definition_import_does_not_load_numerical_sdks():
    _isolated_imports(
        """
from supermimic.world_model.reference import ReferenceFrame, reference_at
reference = ReferenceFrame(None, None, None, None, 'root position', 'root quaternion')
assert reference.root_pos_w == 'root position'
""",
        ["torch", "numpy", "mujoco", "mjlab", "warp", "scipy"],
    )


def test_imagination_import_does_not_load_world_model_trainer():
    _isolated_imports(
        """
from supermimic.imagination import rollout
from supermimic.world_model.reference import reference_at
assert rollout.reference_at is reference_at
assert 'supermimic.world_model.wm_trainer' not in sys.modules
""",
        ["supermimic.world_model.wm_trainer", "supermimic.training.distributed"],
    )


def test_legacy_reference_paths_preserve_object_identity():
    from supermimic.world_model.heads import ReferenceFrame as LegacyFrame
    from supermimic.world_model.wm_trainer import reference_at as legacy_reference_at

    assert LegacyFrame is ReferenceFrame
    assert legacy_reference_at is reference_at


def test_reference_gather_preserves_batch_and_quantity_identity():
    frame = [11, 5, 11, 6]
    root = SimpleNamespace(root_pos_w=object(), root_quat_w=object())
    bodies = SimpleNamespace(pos_w=object(), quat_w=object(), lin_vel_w=object(), ang_vel_w=object())
    calls = []

    class Library:
        def ref(self, indices):
            calls.append(("root", indices))
            return root

        def bodies(self, indices):
            calls.append(("bodies", indices))
            return bodies

    result = reference_at(Library(), frame)
    assert [name for name, _ in calls] == ["root", "bodies"]
    assert all(indices is frame for _, indices in calls)
    expected = (
        bodies.pos_w,
        bodies.quat_w,
        bodies.lin_vel_w,
        bodies.ang_vel_w,
        root.root_pos_w,
        root.root_quat_w,
    )
    assert all(
        getattr(result, field.name) is value
        for field, value in zip(fields(ReferenceFrame), expected, strict=True)
    )
