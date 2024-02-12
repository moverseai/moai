import roma
import roma.internal
import torch

__all__ = ['GramSchmidt', 'ExtractSixd']

class ExtractSixd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix[..., :3, :2] # [B, ..., 3, 3] -> [B, ..., 3, 2]

class GramSchmidt(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
        sixd: torch.Tensor, # [B, R, 3, 2]
    ) -> torch.Tensor: # [B, R, 3, 3]
        return roma.special_gramschmidt(sixd.view(sixd.shape[0], -1, 3, 2))