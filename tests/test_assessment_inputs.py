"""Assessment holds verified paths through scoring without needing a simulator."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path, *, legacy=False, completed=True, failure=None):
    import numpy as np
    from test_motion_preparation import _motion

    from supermimic.deployment.trajectory import PHYSICAL_CONTACT_COLUMNS

    motion = tmp_path / "source/clip.npz"
    _motion(motion, frames=3)
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    run.mkdir()
    (bundle / "model.mjb").write_bytes(b"compiled MJB publication")
    (bundle / "policy.pt").write_bytes(b"policy file hashed without loading Torch")
    contract = {
        "checkpoint_sha256": "a" * 64,
        "control_dt": 0.02,
        "physics_dt": 0.005,
        "decimation": 4,
        "files": {name: _digest(bundle / name) for name in ("model.mjb", "policy.pt")},
    }
    (bundle / "contract.json").write_text(json.dumps(contract))
    states = np.zeros((3, 71))
    states[:, 3] = 1.0
    contacts = np.zeros((8, 13))
    contacts[:, 0] = np.arange(1, 9) * 0.005
    arrays = {
        "states": states,
        "actions": np.zeros((2, 29), dtype=np.float32),
        "reference_frame": np.array([1, 2]),
        "physical_contacts": contacts,
    }
    if not legacy:
        arrays["control_dt"] = np.array(0.02)
        arrays["physical_contact_columns"] = np.array(PHYSICAL_CONTACT_COLUMNS)
    np.savez(run / "trajectory.npz", **arrays)
    result = {
        "status": "complete",
        "auto_reset": False,
        "completed_motion": completed,
        "failure": failure,
        "steps": 2,
        "control_dt": 0.02,
        "reference_goal_offset": 1,
        "motion_sha256": _digest(motion),
        "trajectory_sha256": _digest(run / "trajectory.npz"),
        "contract_sha256": _digest(bundle / "contract.json"),
        "checkpoint_sha256": contract["checkpoint_sha256"],
    }
    (run / "result.json").write_text(json.dumps(result))
    return SimpleNamespace(
        bundle=bundle,
        motion=motion,
        run=run,
        output=tmp_path / "reports/not-created/assessment.json",
    )


def _result(case, **changes):
    path = case.run / "result.json"
    result = json.loads(path.read_text())
    result.update(changes)
    path.write_text(json.dumps(result))


def _rewrite_npz(path, **changes):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(changes)
    np.savez(path, **arrays)


def _open(case):
    from supermimic.deployment.assessment import open_assessment_inputs

    return open_assessment_inputs(case.bundle, case.motion, case.run, case.output)


@pytest.mark.parametrize("legacy", [False, True])
def test_assessment_owns_readonly_snapshots_and_records_legacy_clock_evidence(tmp_path, legacy):
    import numpy as np

    case = _case(tmp_path, legacy=legacy)
    trajectory_sha = _digest(case.run / "trajectory.npz")
    with _open(case) as inputs:
        assert inputs.motion_frames == 3
        assert inputs.positions.shape == (3, 30, 3)
        assert inputs.quaternions.shape == (3, 30, 4)
        assert inputs.completed_motion
        assert inputs.motion_path != case.motion
        assert inputs.model_path != case.bundle / "model.mjb"
        assert inputs.model_path.parent.parent == tmp_path
        assert inputs.motion_path.stat().st_mode & 0o222 == 0
        assert inputs.model_path.stat().st_mode & 0o222 == 0
        assert _digest(inputs.motion_path) == _digest(case.motion)
        assert _digest(inputs.model_path) == _digest(case.bundle / "model.mjb")
        assert inputs.trajectory.sha256 == trajectory_sha
        assert inputs.evidence["artifact_identity"]["status"] == "verified"
        assert inputs.evidence["control_dt_source"] == ("fallback" if legacy else "trajectory")
        assert inputs.evidence["contact_layout_verified"] is (not legacy)
        assert inputs.evidence["reference_clock"] == {"source": "result.json", "offset": 1}
        with np.load(inputs.motion_path, allow_pickle=False) as same_reference:
            assert same_reference["joint_pos"].shape == (3, 29)
        owned = (inputs.motion_path, inputs.model_path)
    assert all(not path.exists() for path in owned)
    assert not list(tmp_path.glob(".supermimic-assessment-*"))
    assert not case.output.parent.exists()


def test_reference_arrays_and_later_path_reads_keep_the_same_publication(tmp_path, monkeypatch):
    import numpy as np
    from test_motion_preparation import _motion

    from supermimic.deployment import assessment

    case = _case(tmp_path)
    old_motion_sha = _digest(case.motion)
    replacement = tmp_path / "new-clip.npz"
    _motion(replacement, frames=7)
    replacement_model = tmp_path / "new-model.mjb"
    replacement_model.write_bytes(b"another model publication")
    original_snapshot = assessment.snapshot_file

    def rotate_after_real_snapshot(source, destination, **kwargs):
        digest = original_snapshot(source, destination, **kwargs)
        if source == case.motion:
            replacement.replace(case.motion)
        elif source == case.bundle / "model.mjb":
            replacement_model.replace(source)
        return digest

    monkeypatch.setattr(assessment, "snapshot_file", rotate_after_real_snapshot)
    with _open(case) as inputs:
        assert inputs.positions.shape == (3, 30, 3)
        # This is the pathname passed to tracking_metrics after sole extraction.
        with np.load(inputs.motion_path, allow_pickle=False) as later_reader:
            assert later_reader["joint_pos"].shape == (3, 29)
        assert _digest(inputs.motion_path) == old_motion_sha
        assert inputs.model_path.read_bytes() == b"compiled MJB publication"
        assert inputs.evidence["artifact_identity"]["status"] == "verified"
    assert _digest(case.motion) != old_motion_sha
    with np.load(case.motion, allow_pickle=False) as current:
        assert current["joint_pos"].shape == (7, 29)


@pytest.mark.parametrize(
    "reason",
    ["consumer", "motion", "model", "policy", "trajectory", "clock", "contacts", "dt", "schema"],
)
def test_assessment_failure_paths_leave_no_owned_snapshot_directory(tmp_path, reason):
    import numpy as np

    case = _case(tmp_path)
    if reason == "motion":
        _result(case, motion_sha256="b" * 64)
    elif reason == "model":
        (case.bundle / "model.mjb").write_bytes(b"changed binary")
    elif reason == "policy":
        (case.bundle / "policy.pt").write_bytes(b"changed policy")
    elif reason == "trajectory":
        _result(case, trajectory_sha256="b" * 64)
    elif reason in ("clock", "contacts", "dt"):
        changes = {
            "clock": {"reference_frame": np.array([1, 1])},
            "contacts": {"physical_contacts": np.zeros((10, 13))},
            "dt": {"control_dt": np.array(0.01)},
        }[reason]
        _rewrite_npz(case.run / "trajectory.npz", **changes)
        _result(case, trajectory_sha256=_digest(case.run / "trajectory.npz"))
    elif reason == "schema":
        with np.load(case.motion, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if key != "joint_vel"}
        np.savez(case.motion, **arrays)
        _result(case, motion_sha256=_digest(case.motion))
    error = RuntimeError if reason == "consumer" else ValueError
    with pytest.raises(error):
        with _open(case):
            assert reason == "consumer"
            raise RuntimeError("scoring failed after reading the inputs")
    assert not list(tmp_path.glob(".supermimic-assessment-*"))
    assert not case.output.exists()


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"auto_reset": True}])
def test_assessment_retains_finished_uninterrupted_admission(tmp_path, changes):
    case = _case(tmp_path)
    _result(case, **changes)
    with pytest.raises(ValueError, match="finished uninterrupted"):
        with _open(case):
            pytest.fail("Invalid rollout protocol must not enter scoring")
    assert not list(tmp_path.glob(".supermimic-assessment-*"))


def test_invalid_source_is_rejected_before_any_snapshot_is_created(tmp_path, monkeypatch):
    import numpy as np

    from supermimic.deployment import assessment

    case = _case(tmp_path)
    with np.load(case.motion, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "joint_vel"}
    np.savez(case.motion, **arrays)
    _result(case, motion_sha256=_digest(case.motion))

    def unexpected_snapshot(*args, **kwargs):
        pytest.fail("Motion schema admission must precede every input snapshot")

    monkeypatch.setattr(assessment, "snapshot_file", unexpected_snapshot)
    with pytest.raises(ValueError, match="Missing motion fields"):
        with _open(case):
            pytest.fail("Invalid source must not enter assessment")
    assert not list(tmp_path.glob(".supermimic-assessment-*"))


@pytest.mark.parametrize("completed,failure", [(False, None), (True, "root_height")])
def test_public_assessor_judges_normal_skill_failure_and_passes_owned_paths(
    tmp_path, monkeypatch, completed, failure
):
    import numpy as np
    from test_skill_acceptance import good_metrics

    from supermimic import cpu_resources
    from supermimic.deployment import metrics

    case = _case(tmp_path, legacy=True, completed=completed, failure=failure)
    script = Path(__file__).resolve().parents[1] / "scripts/assess_skill.py"
    spec = importlib.util.spec_from_file_location("owned_assessment_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_paths = []

    def load_model(path):
        path = Path(path)
        loaded_paths.append(path)
        assert path != case.bundle / "model.mjb"
        assert path.read_bytes() == b"compiled MJB publication"
        return SimpleNamespace(
            ngeom=2,
            geom_bodyid=np.array([0, 1]),
            geom_contype=np.array([1, 1]),
            geom_pos=np.zeros((2, 3)),
            geom_type=np.array([1, 1]),
            geom_size=np.zeros((2, 3)),
            body=lambda name: SimpleNamespace(id=0 if "left" in name else 1),
        )

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjModel = SimpleNamespace(from_binary_path=load_model)
    fake_mujoco.mjtGeom = SimpleNamespace(mjGEOM_SPHERE=1, mjGEOM_CAPSULE=2)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    def score(states, actions, frames, motion, contract):
        loaded_paths.append(motion)
        assert motion != case.motion and motion.stat().st_mode & 0o222 == 0
        with np.load(motion, allow_pickle=False) as reference:
            assert reference["joint_pos"].shape == (3, 29)
        return good_metrics()

    # Setting the module dict avoids triggering its lazy Torch scorer import.
    monkeypatch.setitem(metrics.__dict__, "tracking_metrics", score)
    monkeypatch.setattr(cpu_resources, "cpu_guard", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assess_skill.py",
            "--bundle",
            str(case.bundle),
            "--motion",
            str(case.motion),
            "--run",
            str(case.run),
            "--skill",
            "dance",
            "--output",
            str(case.output),
        ],
    )
    module.main()
    report = json.loads(case.output.read_text())
    assert report["judgement"]["passed"] is False
    assert report["judgement"]["checks"][0] == {
        "name": "completed_motion",
        "value": False,
        "passed": False,
    }
    assert report["control_dt_source"] == "fallback"
    assert report["contact_layout_verified"] is False
    assert len(loaded_paths) == 2 and all(not path.exists() for path in loaded_paths)
    assert not list(tmp_path.glob(".supermimic-assessment-*"))


def test_assessment_input_context_can_run_with_all_simulator_and_policy_imports_blocked(tmp_path):
    case = _case(tmp_path, legacy=True)
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "mujoco", "mujoco_warp", "mjlab", "warp", "scipy"}

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError(f"Assessment preparation imported runtime: {fullname}")

sys.meta_path.insert(0, NoRuntime())
from supermimic.deployment.assessment import open_assessment_inputs

with open_assessment_inputs(*map(Path, sys.argv[2:])) as inputs:
    assert inputs.positions.shape == (3, 30, 3)
    assert inputs.trajectory.control_dt == 0.02
    assert inputs.evidence["control_dt_source"] == "fallback"
    assert inputs.motion_path.exists() and inputs.model_path.exists()
    motion, model = inputs.motion_path, inputs.model_path
assert not motion.exists() and not model.exists()
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(source),
            str(case.bundle),
            str(case.motion),
            str(case.run),
            str(case.output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
