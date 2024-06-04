import roma
import torch

from moai.supervision.losses.regression.cosine_distance import _acos_safe

__all__ = [
    "RotationMatrixAngle",
    "RotationMatrixGeodesic",
]


class RotationMatrixAngle(torch.nn.Module):
    def __init__(
        self,
        safe: bool = False,
    ) -> None:
        super().__init__()
        self.safe = safe

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        if self.safe:
            R = pred.transpose(-1, -2) @ gt
            cos = roma.rotmat_cosine_angle(R)
            return _acos_safe(cos)
        else:
            return roma.rotmat_geodesic_distance_naive(pred, gt)


class RotationMatrixGeodesic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        return roma.rotmat_geodesic_distance(pred, gt)
