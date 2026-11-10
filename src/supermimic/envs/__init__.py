"""Load the simulation layer only when an environment is actually requested."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermimic.envs.tracking_env_cfg import (
        SuperMimicEnvCfgArgs as SuperMimicEnvCfgArgs,
    )
    from supermimic.envs.tracking_env_cfg import (
        make_env as make_env,
    )
    from supermimic.envs.tracking_env_cfg import (
        make_env_cfg as make_env_cfg,
    )

__all__ = ["SuperMimicEnvCfgArgs", "make_env", "make_env_cfg"]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module("supermimic.envs.tracking_env_cfg"), name)
    raise AttributeError(name)
