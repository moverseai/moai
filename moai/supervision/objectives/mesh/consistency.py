import logging
import typing

import torch

from moai.utils.arguments.choices import assert_choices

__all__ = ["SurfaceNormalConsistency", "FaceAreaConsistency"]

log = logging.getLogger(__name__)

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


class SurfaceFeatureConsistency(torch.nn.Module):
    __ORDERS__ = ["inf", 0, 1, 2]

    def __init__(self, order: typing.Union[str, int]):
        super().__init__()
        assert_choices(log, "order", order, SurfaceFeatureConsistency.__ORDERS__)
        self.order = order

    def forward(
        self,
        vertex_features: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: only tested w/ shared faces tensor
        faces = faces.squeeze()
        features_faces = vertex_features[:, faces]
        a = torch.linalg.norm(
            features_faces[..., 0, :] - features_faces[..., 1, :],
            ord=self.order,
            dim=-1,
        )
        b = torch.linalg.norm(
            features_faces[..., 0, :] - features_faces[..., 2, :],
            ord=self.order,
            dim=-1,
        )
        c = torch.linalg.norm(
            features_faces[..., 2, :] - features_faces[..., 1, :],
            ord=self.order,
            dim=-1,
        )
        return torch.sum(torch.cat([a, b, c], dim=-1), dim=-1)


class FaceAreaConsistency(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        deformed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(original / deformed + deformed / original)
