"""Differentiable batched forward kinematics for the Unitree G1 (torch).

Built from a compiled ``mujoco.MjModel`` of ``g1.xml`` so the tree, joint axes,
anchors and ``qpos0`` are exactly MuJoCo's. Reproduces, for every body:

* ``xpos`` / ``xquat``                      (link frame origin pose, world)
* link-frame-origin linear velocity, world  (== mjlab ``body_link_lin_vel_w``)
* angular velocity, world                   (== mjlab ``body_link_ang_vel_w``)

given ``qpos`` (36: root pos, root quat wxyz, 29 hinges) and ``qvel`` (35: root
linear velocity in world, root angular velocity in **body** frame, 29 hinge rates),
i.e. MuJoCo's free-joint conventions.

Quaternion math is element-wise (no 3x3 matmuls) so results are unaffected by TF32.
Position FK follows ``mj_kinematics`` (hinge rotates about its anchor by
``qpos - qpos0``); velocities are propagated down the tree:

    w_b = w_parent + axis_w * qdot
    v_b = v_parent + w_parent x (p_b - p_parent) + (axis_w * qdot) x (p_b - anchor_w)

which equals MuJoCo's ``cvel`` transported to the body origin (``mj_objectVelocity``
with ``flg_local=0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from supermimic.kinematics.rotations import (
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    quat_normalize,
)


@dataclass
class _BodyLink:
    body_id: int  # MuJoCo body id (1..nbody-1)
    parent_index: int  # index into our body list (0..nb-1), -1 for the root
    joint_type: int  # mjJNT_FREE / mjJNT_HINGE / -1 (fixed)
    qpos_adr: int
    dof_adr: int


class G1Kinematics:
    """Batched differentiable FK for a MuJoCo single-tree humanoid (G1)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.nbody = model.nbody - 1  # exclude world
        self.nq = model.nq
        self.nv = model.nv
        self.body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)
        ]
        self.joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
        ]

        links: list[_BodyLink] = []
        for b in range(1, model.nbody):
            parent = int(model.body_parentid[b])
            parent_index = -1 if parent == 0 else parent - 1
            assert parent < b, "MuJoCo bodies are topologically ordered"
            njnt = int(model.body_jntnum[b])
            if njnt == 0:
                links.append(_BodyLink(b, parent_index, -1, -1, -1))
                continue
            assert njnt == 1, f"body {b} has {njnt} joints; only 0/1 supported"
            j = int(model.body_jntadr[b])
            jt = int(model.jnt_type[j])
            assert jt in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE), (
                f"joint type {jt}"
            )
            links.append(
                _BodyLink(b, parent_index, jt, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
            )
        assert links[0].joint_type == mujoco.mjtJoint.mjJNT_FREE and links[0].parent_index == -1
        self.links = links

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)  # noqa: E731
        self.body_pos = t(model.body_pos[1:])  # (nb,3)
        self.body_quat = t(model.body_quat[1:])  # (nb,4) wxyz
        jnt_pos = np.zeros((self.nbody, 3))
        jnt_axis = np.zeros((self.nbody, 3))
        qpos0 = np.zeros(self.nbody)
        for i, link in enumerate(links):
            if link.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                j = int(model.body_jntadr[link.body_id])
                jnt_pos[i] = model.jnt_pos[j]
                jnt_axis[i] = model.jnt_axis[j]
                qpos0[i] = model.qpos0[link.qpos_adr]
        self.jnt_pos = t(jnt_pos)
        self.jnt_axis = t(jnt_axis)
        self.qpos0 = t(qpos0)
        # Hinge bookkeeping in body order (skip root / fixed bodies).
        self.hinge_body_idx = [
            i for i, link in enumerate(links) if link.joint_type == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.hinge_qpos_adr = torch.tensor(
            [links[i].qpos_adr for i in self.hinge_body_idx], device=self.device
        )
        self.hinge_dof_adr = torch.tensor(
            [links[i].dof_adr for i in self.hinge_body_idx], device=self.device
        )
        # Bodies at the same depth are independent. Evaluate all limbs at that
        # depth together rather than launching a sequence of tiny kernels per body.
        depths = [0] * self.nbody
        for i, link in enumerate(links[1:], 1):
            depths[i] = depths[link.parent_index] + 1
        groups = [
            [i for i, depth in enumerate(depths) if depth == d] for d in range(max(depths) + 1)
        ]
        self.levels = []
        order = [0]
        for previous, group in zip(groups[:-1], groups[1:], strict=True):
            ids = torch.tensor(group, device=self.device)
            parents = torch.tensor(
                [previous.index(links[i].parent_index) for i in group], device=self.device
            )
            qids = torch.tensor([max(0, links[i].qpos_adr) for i in group], device=self.device)
            vids = torch.tensor([max(0, links[i].dof_adr) for i in group], device=self.device)
            hinge = t([float(links[i].joint_type == mujoco.mjtJoint.mjJNT_HINGE) for i in group])
            self.levels.append(
                (
                    parents,
                    qids,
                    vids,
                    hinge,
                    self.body_pos[ids],
                    self.body_quat[ids],
                    self.jnt_pos[ids],
                    self.jnt_axis[ids],
                    self.qpos0[ids],
                )
            )
            order.extend(group)
        self.body_order = torch.tensor(np.argsort(order), device=self.device)

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def body_indices(self, names) -> torch.Tensor:
        return torch.tensor([self.body_names.index(n) for n in names], device=self.device)

    # ------------------------------------------------------------------ core

    def forward_serial(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None):
        """FK for a batch.

        Args:
          qpos: (..., nq) MuJoCo generalized positions.
          qvel: (..., nv) MuJoCo generalized velocities (free joint: lin world, ang body).
        Returns:
          dict with 'pos' (..., nb, 3), 'quat' (..., nb, 4) and, if qvel is given,
          'lin_vel' (..., nb, 3), 'ang_vel' (..., nb, 3) in world frame at the link origin.
        """
        batch = qpos.shape[:-1]
        nb = self.nbody
        pos = [None] * nb
        quat = [None] * nb
        lin = [None] * nb
        ang = [None] * nb
        compute_vel = qvel is not None

        # Root (free joint).
        root_pos = qpos[..., 0:3]
        root_quat = quat_normalize(qpos[..., 3:7])
        pos[0] = root_pos
        quat[0] = root_quat
        if compute_vel:
            lin[0] = qvel[..., 0:3]
            ang[0] = quat_apply(root_quat, qvel[..., 3:6])  # body -> world

        for i in range(1, nb):
            link = self.links[i]
            p = link.parent_index
            pq = quat[p]
            pp = pos[p]
            # Body frame from parent frame (fixed offset).
            b_pos = pp + quat_apply(pq, self.body_pos[i].expand(*batch, 3))
            b_quat = quat_mul(pq, self.body_quat[i].expand(*batch, 4))
            if link.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                anchor = b_pos + quat_apply(b_quat, self.jnt_pos[i].expand(*batch, 3))
                axis_w = quat_apply(b_quat, self.jnt_axis[i].expand(*batch, 3))
                angle = qpos[..., link.qpos_adr] - self.qpos0[i]
                q_loc = quat_from_angle_axis(angle, self.jnt_axis[i].expand(*batch, 3))
                b_quat = quat_mul(b_quat, q_loc)
                b_pos = anchor - quat_apply(b_quat, self.jnt_pos[i].expand(*batch, 3))
                if compute_vel:
                    qd = qvel[..., link.dof_adr].unsqueeze(-1)
                    w_joint = axis_w * qd
                    ang[i] = ang[p] + w_joint
                    lin[i] = (
                        lin[p]
                        + torch.cross(ang[p], b_pos - pp, dim=-1)
                        + torch.cross(w_joint, b_pos - anchor, dim=-1)
                    )
            else:  # fixed body
                if compute_vel:
                    ang[i] = ang[p]
                    lin[i] = lin[p] + torch.cross(ang[p], b_pos - pp, dim=-1)
            pos[i] = b_pos
            quat[i] = b_quat

        out = {"pos": torch.stack(pos, dim=-2), "quat": torch.stack(quat, dim=-2)}
        if compute_vel:
            out["lin_vel"] = torch.stack(lin, dim=-2)
            out["ang_vel"] = torch.stack(ang, dim=-2)
        return out

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None):
        """Depth-batched FK; same body ordering and autograd semantics as the serial reference."""
        batch = qpos.shape[:-1]
        positions = [qpos[..., None, :3]]
        quaternions = [quat_normalize(qpos[..., None, 3:7])]
        linear = angular = None
        if qvel is not None:
            linear = [qvel[..., None, :3]]
            angular = [quat_apply(quaternions[0], qvel[..., None, 3:6])]
        for parents, qids, vids, hinge, bpos, bquat, jpos, axis, qzero in self.levels:
            bpos = bpos.expand(*batch, -1, 3)
            bquat = bquat.expand(*batch, -1, 4)
            jpos = jpos.expand(*batch, -1, 3)
            axis = axis.expand(*batch, -1, 3)
            parent_pos = positions[-1].index_select(-2, parents)
            parent_quat = quaternions[-1].index_select(-2, parents)
            pos = parent_pos + quat_apply(parent_quat, bpos)
            quat = quat_mul(parent_quat, bquat)
            anchor = pos + quat_apply(quat, jpos)
            axis_w = quat_apply(quat, axis)
            angle = (qpos.index_select(-1, qids) - qzero) * hinge
            quat = quat_mul(quat, quat_from_angle_axis(angle, axis))
            pos = anchor - quat_apply(quat, jpos)
            if qvel is not None:
                parent_ang = angular[-1].index_select(-2, parents)
                parent_lin = linear[-1].index_select(-2, parents)
                joint_ang = axis_w * (qvel.index_select(-1, vids) * hinge)[..., None]
                angular.append(parent_ang + joint_ang)
                linear.append(
                    parent_lin
                    + torch.cross(parent_ang, pos - parent_pos, dim=-1)
                    + torch.cross(joint_ang, pos - anchor, dim=-1)
                )
            positions.append(pos)
            quaternions.append(quat)
        result = {
            "pos": torch.cat(positions, dim=-2).index_select(-2, self.body_order),
            "quat": torch.cat(quaternions, dim=-2).index_select(-2, self.body_order),
        }
        if qvel is not None:
            result.update(
                lin_vel=torch.cat(linear, dim=-2).index_select(-2, self.body_order),
                ang_vel=torch.cat(angular, dim=-2).index_select(-2, self.body_order),
            )
        return result

    __call__ = forward
