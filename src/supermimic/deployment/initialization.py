"""Declared, bounded initial-state changes for reproducible simulation checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number within [{low}, {high}]")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number within [{low}, {high}]")
    return value


def _vector(value, name, bound):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(_number(v, name, -bound, bound) for v in value)


@dataclass(frozen=True)
class InitialPerturbation:
    """Only the initial state changes; policy, PD and model parameters stay fixed.

    Root translation and linear velocity use world coordinates. The rotation
    vector is left-multiplied in the world frame; angular velocity uses the
    free joint's body coordinates. Joint noise uses a private PCG64 generator
    and the contract's initial soft limits. Bounds are protocol input limits,
    not quality acceptance thresholds or guarantees of recovery.
    """

    root_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_world_rotation_vector_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_linear_velocity_delta_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_body_angular_velocity_delta_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_position_noise_std_rad: float = 0.0
    seed: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping) -> InitialPerturbation:
        if not isinstance(value, Mapping) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Initial perturbation contains unknown fields or is not an object")
        defaults = asdict(cls())
        defaults.update(value)
        bounds = {
            "root_translation_m": 0.1,
            "root_world_rotation_vector_rad": 0.3,
            "root_linear_velocity_delta_m_s": 1.0,
            "root_body_angular_velocity_delta_rad_s": 2.0,
        }
        for name, bound in bounds.items():
            defaults[name] = _vector(defaults[name], name, bound)
        name = "joint_position_noise_std_rad"
        defaults[name] = _number(defaults[name], name, 0.0, 0.1)
        seed = defaults["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Initialization seed must be an integer within [0, 2**32 - 1]")
        return cls(**defaults)

    def as_dict(self) -> dict:
        return asdict(self)

    def apply(self, data, contract: dict) -> None:
        from supermimic.deployment.reference import rotation

        # Validate even instances created directly rather than through the parser.
        value = self.from_mapping(self.as_dict())
        rq, rv = contract["root_qpos_index"], contract["root_qvel_index"]
        if any(value.root_translation_m):
            data.qpos[rq : rq + 3] += value.root_translation_m
        if any(value.root_world_rotation_vector_rad):
            from scipy.spatial.transform import Rotation

            orientation = Rotation.from_rotvec(value.root_world_rotation_vector_rad) * rotation(
                data.qpos[rq + 3 : rq + 7]
            )
            data.qpos[rq + 3 : rq + 7] = orientation.as_quat()[[3, 0, 1, 2]]
        if any(value.root_linear_velocity_delta_m_s):
            data.qvel[rv : rv + 3] += value.root_linear_velocity_delta_m_s
        if any(value.root_body_angular_velocity_delta_rad_s):
            data.qvel[rv + 3 : rv + 6] += value.root_body_angular_velocity_delta_rad_s
        if value.joint_position_noise_std_rad:
            indices = contract["joint_qpos_indices"]
            limits = np.asarray(contract["joint_limits"])
            center = limits.mean(axis=1)
            half = (
                (limits[:, 1] - limits[:, 0])
                * contract.get("initial_joint_soft_limit_factor", 0.9)
                / 2
            )
            noise = np.random.Generator(np.random.PCG64(value.seed)).normal(
                0, value.joint_position_noise_std_rad, len(indices)
            )
            data.qpos[indices] = np.clip(data.qpos[indices] + noise, center - half, center + half)


def parse_initialization_plan(value: dict) -> list[tuple[str, InitialPerturbation]]:
    """Admit a small, ordered suite with a mandatory exact zero control first."""
    if not isinstance(value, dict) or set(value) != {"schema", "cases"}:
        raise ValueError("Initialization plan requires only schema and cases")
    if value["schema"] != "g1-static-initialization-v1":
        raise ValueError("Unsupported initialization plan schema")
    cases = value["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("Initialization plan requires 1 to 64 cases")
    result, names = [], set()
    for row in cases:
        if not isinstance(row, dict) or set(row) != {"id", "perturbation"}:
            raise ValueError("Each initialization case requires only id and perturbation")
        name = row["id"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name) is None
        ):
            raise ValueError("Initialization case id must be a safe, short directory name")
        if name in names:
            raise ValueError("Initialization case ids must be unique")
        names.add(name)
        result.append((name, InitialPerturbation.from_mapping(row["perturbation"])))
    if result[0] != ("baseline", InitialPerturbation()):
        raise ValueError("The first initialization case must be the zero baseline")
    return result
