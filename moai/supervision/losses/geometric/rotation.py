import torch
import roma

__all__ = [
    'RotationAngle',
    'RotationGeodesic',
]

class RotationAngle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
        pred:   torch.Tensor,
        gt:     torch.Tensor,        
    ) -> torch.Tensor:
        return roma.rotmat_geodesic_distance_naive(pred, gt)

class RotationGeodesic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
        pred:   torch.Tensor,
        gt:     torch.Tensor,        
    ) -> torch.Tensor:
        return roma.rotmat_geodesic_distance(pred, gt)