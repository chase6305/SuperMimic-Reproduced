"""Planning and schema inspection must work before tensor/simulator SDK imports."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from supermimic import config


def test_profiles_and_memory_planning_do_not_import_tensor_or_simulator_sdks():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "mujoco", "mujoco_warp", "mjlab", "warp", "numpy", "scipy"}

class NoSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError(f"Planning imported SDK: {fullname}")

sys.meta_path.insert(0, NoSDKs())
from supermimic.config import TrainConfig
from supermimic.memory import estimate_training_memory
from supermimic.profiles import training_config

for profile in ("smoke", "g1", "paper-width"):
    cfg = training_config(profile, Path("motions"), Path("output"))
    assert isinstance(cfg, TrainConfig)
    assert estimate_training_memory(cfg, 100)["estimated_cuda_mib"] > 0
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("supermimic.policy.nets", "PolicyCfg"),
        ("supermimic.policy.nets", "CriticCfg"),
        ("supermimic.imagination.policy_trainer", "PolicyTrainerCfg"),
        ("supermimic.world_model.state_spec", "StateSpecCfg"),
        ("supermimic.world_model.world_model", "WorldModelCfg"),
        ("supermimic.world_model.wm_trainer", "WMTrainerCfg"),
    ],
)
def test_configuration_classes_keep_legacy_import_paths(module, name):
    assert getattr(importlib.import_module(module), name) is getattr(config, name)
