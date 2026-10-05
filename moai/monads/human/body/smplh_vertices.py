"""Temprary copy of SMPLH vertices."""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["SMPLHVertices"]


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3), Rodrigues; exact identity at zero angle."""
    theta = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(
        *axis.shape[:-1], 3, 3
    )
    sin, cos = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + sin * K + (1.0 - cos) * (K @ K)


class SMPLHVertices(nn.Module):
    """forward(betas, pose, translation) -> (B, L, 3) world-space vertex positions for the
    configured `vertex_subset` (L = len(vertex_subset))."""

    def __init__(
        self,
        model_path: str,
        vertex_subset: Sequence[int],
        n_betas: int = 16,
    ) -> None:
        super().__init__()
        data = dict(np.load(model_path, allow_pickle=True))
        v_template = data["v_template"].astype(np.float32)  # (V, 3)
        shapedirs = data["shapedirs"].astype(np.float32)  # (V, 3, n_betas_full)
        J_regressor = data["J_regressor"].astype(np.float32)  # (J, V)
        joints_template = J_regressor @ v_template  # (J, 3)
        joints_shapedirs = np.einsum(
            "jv,vdb->jdb", J_regressor, shapedirs
        )  # (J, 3, n_betas_full)

        subset = np.asarray(vertex_subset, np.int64)
        parents = data["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        self.parents = parents.tolist()
        self.n_joints = J_regressor.shape[0]
        self.n_betas = n_betas

        self.register_buffer(
            "v_template", torch.as_tensor(v_template[subset])
        )  # (L, 3)
        self.register_buffer(
            "shapedirs", torch.as_tensor(shapedirs[subset, :, :n_betas])
        )  # (L,3,B)
        self.register_buffer(
            "posedirs", torch.as_tensor(data["posedirs"].astype(np.float32)[subset])
        )  # (L,3,(J-1)*9)
        self.register_buffer(
            "weights", torch.as_tensor(data["weights"].astype(np.float32)[subset])
        )  # (L, J)
        self.register_buffer(
            "joints_template", torch.as_tensor(joints_template)
        )  # (J, 3)
        self.register_buffer(
            "joints_shapedirs", torch.as_tensor(joints_shapedirs[..., :n_betas])
        )  # (J,3,B)

    def forward(
        self, betas: torch.Tensor, pose: torch.Tensor, translation: torch.Tensor
    ) -> torch.Tensor:
        """betas (B, n_betas), pose (B, n_joints, 3) axis-angle root-first, translation (B, 3)
        -> (B, L, 3) world-space vertex positions."""
        b = betas.shape[0]
        betas = betas[:, : self.n_betas]
        v_shaped = self.v_template[None] + torch.einsum(
            "ldb,nb->nld", self.shapedirs, betas
        )
        joints_rest = self.joints_template[None] + torch.einsum(
            "jdb,nb->njd", self.joints_shapedirs, betas
        )

        rotmats = axis_angle_to_matrix(pose)  # (B, J, 3, 3)
        eye3 = torch.eye(3, device=pose.device, dtype=pose.dtype)
        pose_feature = (rotmats[:, 1:] - eye3).reshape(b, -1)
        v_posed = v_shaped + torch.einsum("lde,ne->nld", self.posedirs, pose_feature)

        # local -> global rigid transforms per joint, built purely out-of-place (list + stack, no
        # indexed in-place writes) so this stays vmap-compatible: theseus's AutoDiffCostFunction
        # computes jacobians via vmap(jacrev(...)), which in-place tensor mutation breaks.
        local_t = [joints_rest[:, 0]]
        for j in range(1, self.n_joints):
            parent = self.parents[j]
            local_t.append(joints_rest[:, j] - joints_rest[:, parent])
        local_t = torch.stack(local_t, 1)  # (B, J, 3)

        global_r = [rotmats[:, 0]]
        global_t = [local_t[:, 0]]
        for j in range(1, self.n_joints):
            parent = self.parents[j]
            global_r.append(global_r[parent] @ rotmats[:, j])
            global_t.append(
                global_t[parent] + (global_r[parent] @ local_t[:, j, :, None])[..., 0]
            )
        global_r = torch.stack(global_r, 1)  # (B, J, 3, 3)
        global_t = torch.stack(global_t, 1)  # (B, J, 3)

        # inverse-bind step (rest-pose joint location, rotated, subtracted out) turns this into a
        # pure per-vertex skinning transform; since the homogeneous w-coordinate is always 1,
        # applying it is just rotate-then-translate, no need for explicit 4x4 matrices
        skin_t = global_t - torch.einsum("njkl,njl->njk", global_r, joints_rest)
        per_joint = (
            torch.einsum("njkl,nvl->nvjk", global_r, v_posed) + skin_t[:, None]
        )  # (B,L,J,3)
        vertices = torch.einsum("vj,nvjk->nvk", self.weights, per_joint)
        return vertices + translation[:, None]
