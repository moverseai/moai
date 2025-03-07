# NOTE: adapted from pytorch3d

import torch
from pytorch3d.ops import cot_laplacian, laplacian

__all__ = ["MeshLaplacianSmoothless"]

r"""
Computes the laplacian smoothing objective for a batch of meshes.
This function supports three variants of Laplacian smoothing,
namely with uniform weights("uniform"), with cotangent weights ("cot"),
and cotangent curvature ("cotcurv").For more details read [1, 2].

Args:
    meshes: Meshes object with a batch of meshes.
    method: str specifying the method for the laplacian.
Returns:
    loss: Average laplacian smoothing loss across the batch.
    Returns 0 if meshes contains no meshes or all empty meshes.

Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
the surface normal, while the curvature variant LckV[i] scales the normals
by the discrete mean curvature. For vertex i, assume S[i] is the set of
neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
two triangles connecting vertex v_i and its neighboring vertex v_j
for j in S[i], as seen in the diagram below.

.. code-block:: python

            a_ij
            /\
            /  \
            /    \
            /      \
    v_i /________\ v_j
        \        /
            \      /
            \    /
            \  /
            \/
            b_ij

    The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
    For the uniform variant,    w_ij = 1 / |S[i]|
    For the cotangent variant,
        w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
    For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
    where A[i] is the sum of the areas of all triangles containing vertex v_i.

There is a nice trigonometry identity to compute cotangents. Consider a triangle
with side lengths A, B, C and angles a, b, c.

.. code-block:: python

            c
            /|\
            / | \
        /  |  \
        B /  H|   \ A
        /    |    \
        /     |     \
    /a_____|_____b\
            C

    Then cot a = (B^2 + C^2 - A^2) / 4 * area
    We know that area = CH/2, and by the law of cosines we have

    A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

    Putting these together, we get:

    B^2 + C^2 - A^2     2BC cos a
    _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
        4 * area            2CH


[1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
and curvature flow", SIGGRAPH 1999.

[2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
"""

# TODO: see https://github.com/nmwsharp/robust-laplacians-py


class MeshLaplacianSmoothless(torch.nn.Module):
    def __init__(self, method: str = "uniform"):
        super().__init__()
        self.method = method

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        vertices: torch.Tensor,  # [B, V, 3]
        faces: torch.Tensor,  # [B, F, 3]
    ) -> torch.Tensor:
        assert len(vertices.shape) == 3 and len(faces.shape) >= 2
        assert faces.dtype == torch.int64
        N = vertices.shape[0]
        # if faces.dtype != torch.int64:
        #     faces = faces.to(torch.int64)
        if len(faces.shape) == 2:
            faces = faces.unsqueeze(0).expand(N, *faces.shape)  # assume shared faces
        total_loss = torch.scalar_tensor(
            0.0, dtype=vertices.dtype, device=vertices.device
        )
        for verts_packed, faces_packed in zip(vertices, faces):
            weights = 1.0 / vertices.shape[1]  # TODO: revisit this
            # We don't want to backprop through the computation of the Laplacian;
            # just treat it as a magic constant matrix that is used to transform
            # verts into normals
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                if self.method == "uniform":
                    L = laplacian(verts_packed.float(), faces_packed)
                elif self.method in ["cot", "cotcurv"]:
                    L, inv_areas = cot_laplacian(verts_packed, faces_packed)
                    if self.method == "cot":
                        norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                        idx = norm_w > 0
                        # pyre-fixme[58]: `/` is not supported for operand types `float` and
                        #  `Tensor`.
                        norm_w[idx] = 1.0 / norm_w[idx]
                    else:
                        L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                        norm_w = 0.25 * inv_areas
                else:
                    raise ValueError("Method should be one of {uniform, cot, cotcurv}")

            if self.method == "uniform":
                loss = L.mm(verts_packed)
            elif self.method == "cot":
                # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
                loss = L.mm(verts_packed) * norm_w - verts_packed
            elif self.method == "cotcurv":
                # pyre-fixme[61]: `norm_w` may not be initialized here.
                loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
            loss = loss.norm(dim=1)
            loss = loss * weights
            total_loss = total_loss + loss
        return total_loss.sum() / N
