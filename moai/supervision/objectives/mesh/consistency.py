import torch

__all__ = ["SurfaceNormalConsistency", "FaceAreaConsistency"]

# NOTE: from mesh strikes back


class SurfaceNormalConsistency(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        vertex_normals: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: only tested w/ shared faces tensor
        faces = faces.squeeze()
        normals_faces = vertex_normals[:, faces]
        # a = torch.sum(normals_faces[..., 0] * normals_faces[..., 1], dim=-1)
        # b = torch.sum(normals_faces[..., 0] * normals_faces[..., 2], dim=-1)
        # c = torch.sum(normals_faces[..., 2] * normals_faces[..., 1], dim=-1)
        a = torch.sum(normals_faces[..., 0, :] * normals_faces[..., 1, :], dim=-1)
        b = torch.sum(normals_faces[..., 0, :] * normals_faces[..., 2, :], dim=-1)
        c = torch.sum(normals_faces[..., 2, :] * normals_faces[..., 1, :], dim=-1)
        return torch.sum(1.0 - torch.stack([a, b, c]), dim=0)


class FaceAreaConsistency(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        deformed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(original / deformed + deformed / original)
