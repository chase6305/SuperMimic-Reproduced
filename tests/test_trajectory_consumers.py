"""Offline consumers keep their own input policies and report evidence honestly."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot_module():
    path = Path(__file__).resolve().parents[1] / "scripts/plot_deployment.py"
    spec = importlib.util.spec_from_file_location("trajectory_plot_consumer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, *, labels=True, result=False):
    import numpy as np

    from supermimic.deployment.trajectory import PHYSICAL_CONTACT_COLUMNS
    from supermimic.interfaces import G1_JOINT_NAMES

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    trajectory, motion = tmp_path / "trajectory.npz", tmp_path / "motion.npz"
    contract = {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_iteration": 7,
        "control_dt": 0.02,
        "first_goal_frame": 1,
        "initial_reference_state_frame": 0,
        "root_qpos_index": 0,
        "joint_names": list(G1_JOINT_NAMES),
        "joint_qpos_indices": list(range(7, 36)),
    }
    (bundle / "contract.json").write_text(json.dumps(contract))
    states = np.zeros((3, 71))
    states[:, 2] = 1.0
    contacts = np.zeros((10, 13))  # Two complete controls and two partial substeps.
    contacts[:, 0] = np.arange(1, 11) * 0.005
    arrays = {
        "states": states,
        "control_dt": np.array(0.02),
        "reference_frame": np.array([1, 2]),
        "physical_contacts": contacts,
    }
    if labels:
        arrays["physical_contact_columns"] = np.array(PHYSICAL_CONTACT_COLUMNS)
    np.savez(trajectory, **arrays)  # Actions are deliberately not a plot requirement.
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 2] = 1.0
    # A legitimate diagnostic reference needs only these three fields, without
    # fps, velocities, quaternions or all 30 training-ABI body names.
    np.savez(
        motion, body_names=np.array(["pelvis"]), body_pos_w=positions, joint_pos=np.zeros((3, 29))
    )
    if result:
        record = {
            "status": "failed",
            "auto_reset": False,
            "completed_motion": False,
            "steps": 2,
            "control_dt": 0.02,
            "reference_goal_offset": 1,
            "trajectory_sha256": _digest(trajectory),
            "motion_sha256": _digest(motion),
            "contract_sha256": _digest(bundle / "contract.json"),
            "checkpoint_sha256": contract["checkpoint_sha256"],
        }
        (tmp_path / "result.json").write_text(json.dumps(record))
    return bundle, trajectory, motion


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_render_accepts_single_states_only_frame_and_legacy_single_dt(tmp_path, shape):
    import numpy as np

    from supermimic.deployment.render import _render_inputs

    bundle, trajectory, _ = _inputs(tmp_path)
    np.savez(trajectory, states=np.array([[1.0, 2.0, 3.0]]), control_dt=np.full(shape, 0.02))
    _, recorded, evidence = _render_inputs(bundle, trajectory)
    assert recorded.states.shape == (1, 3)  # Width belongs to the loaded model.
    assert recorded.control_dt == 0.02
    assert recorded.sha256 == _digest(trajectory)
    assert evidence["artifact_identity"]["status"] == "unverified"
    assert evidence["artifact_identity"]["reason"] == "result.json absent"


def test_plot_reads_minimal_motion_and_failed_partial_trajectory(tmp_path):
    bundle, trajectory, motion = _inputs(tmp_path, result=True)
    data = _plot_module()._plot_inputs(bundle, trajectory, motion)
    assert data["states"].shape == (3, 71)
    assert data["contacts"].shape == (10, 13)
    assert data["reference_position"].shape == (3, 3)
    assert data["reference_joints"].shape == (3, 29)
    assert data["indices"].tolist() == [0, 1, 2]
    assert data["evidence"]["recorded_status"] == "failed"
    assert data["evidence"]["artifact_identity"]["status"] == "verified"
    assert data["evidence"]["physical_contact_columns_verified"] is True


def test_plot_marks_absent_result_and_contact_labels_as_unverified(tmp_path):
    bundle, trajectory, motion = _inputs(tmp_path, labels=False)
    data = _plot_module()._plot_inputs(bundle, trajectory, motion)
    assert data["evidence"]["artifact_identity"]["status"] == "unverified"
    assert data["evidence"]["recorded_status"] is None
    assert data["evidence"]["physical_contact_columns_verified"] is False
    assert data["contacts"].shape == (10, 13)


def test_plot_keeps_unsigned_reference_frames_usable_as_array_indices(tmp_path):
    import numpy as np

    bundle, trajectory, motion = _inputs(tmp_path)
    with np.load(trajectory, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["reference_frame"] = arrays["reference_frame"].astype(np.uint64)
    np.savez(trajectory, **arrays)
    data = _plot_module()._plot_inputs(bundle, trajectory, motion)
    assert data["indices"].dtype == np.intp
    assert data["reference_position"][data["indices"]].shape == (3, 3)


@pytest.mark.parametrize(
    "field", ["trajectory_sha256", "motion_sha256", "contract_sha256", "checkpoint_sha256"]
)
def test_plot_rejects_a_known_identity_mismatch(tmp_path, field):
    bundle, trajectory, motion = _inputs(tmp_path, result=True)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    result[field] = "b" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Artifact identity mismatch: " + field):
        _plot_module()._plot_inputs(bundle, trajectory, motion)


@pytest.mark.parametrize("source", ["bundle", "motion"])
def test_plot_rejects_different_actual_bundle_or_motion(tmp_path, source):
    import numpy as np

    bundle, trajectory, motion = _inputs(tmp_path, result=True)
    if source == "bundle":
        contract_path = bundle / "contract.json"
        contract = json.loads(contract_path.read_text())
        contract["checkpoint_iteration"] = 8
        contract_path.write_text(json.dumps(contract))
    else:
        np.savez(
            motion,
            body_names=np.array(["pelvis"]),
            body_pos_w=np.ones((3, 1, 3)),
            joint_pos=np.zeros((3, 29)),
        )
    with pytest.raises(ValueError, match="Artifact identity mismatch"):
        _plot_module()._plot_inputs(bundle, trajectory, motion)


def test_legacy_missing_fingerprint_is_unverified_but_known_mismatches_still_fail(tmp_path):
    bundle, trajectory, motion = _inputs(tmp_path, result=True)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    del result["contract_sha256"]
    result_path.write_text(json.dumps(result))
    evidence = _plot_module()._plot_inputs(bundle, trajectory, motion)["evidence"]
    assert evidence["artifact_identity"]["status"] == "unverified"
    assert evidence["artifact_identity"]["missing_fields"] == ["contract_sha256"]
    result["motion_sha256"] = "b" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="motion_sha256"):
        _plot_module()._plot_inputs(bundle, trajectory, motion)


def test_plot_motion_replacement_keeps_the_consumed_arrays_and_old_sha(tmp_path, monkeypatch):
    import numpy as np

    from supermimic.deployment import evidence

    bundle, trajectory, motion = _inputs(tmp_path, result=True)
    old_sha = _digest(motion)
    replacement = tmp_path / "new-motion.npz"
    np.savez(
        replacement,
        body_names=np.array(["pelvis"]),
        body_pos_w=np.full((7, 1, 3), 7.0),
        joint_pos=np.full((7, 29), 7.0),
    )
    real_load = np.load
    streams = []

    def rotate_motion(stream, **kwargs):
        if Path(stream.name) == motion:
            replacement.replace(motion)
            streams.append(stream)
        return real_load(stream, **kwargs)

    monkeypatch.setattr(evidence.np, "load", rotate_motion)
    data = _plot_module()._plot_inputs(bundle, trajectory, motion)
    assert data["reference_position"].shape == (3, 3)
    assert data["reference_joints"].shape == (3, 29)
    assert data["evidence"]["motion_sha256"] == old_sha
    assert data["evidence"]["artifact_identity"]["status"] == "verified"
    assert streams[0].closed
    assert _digest(motion) != old_sha


@pytest.mark.parametrize("mutation", ["frames", "dt"])
def test_plot_rejects_clock_mismatch_even_without_result(tmp_path, mutation):
    import numpy as np

    bundle, trajectory, motion = _inputs(tmp_path)
    with np.load(trajectory, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if mutation == "frames":
        arrays["reference_frame"] = np.array([1, 1])
    else:
        arrays["control_dt"] = np.array(0.01)
    np.savez(trajectory, **arrays)
    with pytest.raises(ValueError, match="Reference clock|control_dt"):
        _plot_module()._plot_inputs(bundle, trajectory, motion)


def test_plot_entry_writes_unverified_metadata_for_legacy_diagnostic_input(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from supermimic import cpu_resources

    bundle, trajectory, motion = _inputs(tmp_path, labels=False)
    output = tmp_path / "plot.png"
    module = _plot_module()
    monkeypatch.setattr(cpu_resources, "cpu_guard", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_deployment.py",
            "--bundle",
            str(bundle),
            "--trajectory",
            str(trajectory),
            "--motion",
            str(motion),
            "--output",
            str(output),
        ],
    )
    saved = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        saved["title"] = figure._suptitle.get_text()
        saved["contact_title"] = figure.axes[4].get_title()
        saved["contact_lines"] = len(figure.axes[4].lines)
        return real_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    before = set(plt.get_fignums())
    module.main()
    assert output.is_file()
    record = json.loads(output.with_suffix(".json").read_text())
    assert record["artifact_identity"]["status"] == "unverified"
    assert record["physical_contact_columns_verified"] is False
    assert "unverified" in saved["title"] and "no reset" not in saved["title"]
    assert "unverified" in saved["contact_title"] and saved["contact_lines"] == 0
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("failure", [None, "loader", "fingerprint"])
def test_render_native_loader_receives_owned_verified_model_and_cleans_it(tmp_path, failure):
    from supermimic.deployment.render import _load_render_model

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "model.mjb"
    source.write_bytes(b"original MJB publication")
    replacement = tmp_path / "new-model.mjb"
    replacement.write_bytes(b"new MJB publication")
    contract = {"files": {"model.mjb": "b" * 64 if failure == "fingerprint" else _digest(source)}}
    output = tmp_path / "not-created/subdir/render.mp4"
    opened = []

    def native_loader(path):
        pinned = Path(path)
        opened.append(pinned)
        assert pinned != source
        assert pinned.parent.parent == tmp_path  # Nearest existing output ancestor.
        assert pinned.stat().st_mode & 0o222 == 0
        replacement.replace(source)
        assert pinned.read_bytes() == b"original MJB publication"
        if failure == "loader":
            raise RuntimeError("native load failed")
        return "in-memory model"

    if failure is None:
        assert _load_render_model(bundle, contract, output, native_loader) == "in-memory model"
    else:
        error = RuntimeError if failure == "loader" else ValueError
        with pytest.raises(error):
            _load_render_model(bundle, contract, output, native_loader)
    assert all(not path.exists() and not path.parent.exists() for path in opened)
    assert bool(opened) is (failure != "fingerprint")
    assert not list(tmp_path.glob(".supermimic-render-*"))
    assert not output.parent.exists()


def test_consumer_data_preparation_runs_without_policy_or_simulator_sdks(tmp_path):
    bundle, trajectory, motion = _inputs(tmp_path)
    project = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "mujoco", "mujoco_warp", "mjlab", "warp", "scipy"}

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError(f"Offline data preparation imported runtime: {fullname}")

sys.meta_path.insert(0, NoRuntime())
from supermimic.deployment import evidence, trajectory
from supermimic.deployment.render import _render_inputs

bundle, saved, motion = map(Path, sys.argv[2:5])
_, recorded, _ = _render_inputs(bundle, saved)
assert recorded.states.shape == (3, 71)
spec = importlib.util.spec_from_file_location("isolated_plot", sys.argv[5])
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
data = plot._plot_inputs(bundle, saved, motion)
assert data["reference_position"].shape == (3, 3)
assert data["evidence"]["artifact_identity"]["status"] == "unverified"
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(project / "src"),
            str(bundle),
            str(trajectory),
            str(motion),
            str(project / "scripts/plot_deployment.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
