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
        sixd: torch.Tensor, # [..., R, 3, 2]
    ) -> torch.Tensor: # [..., R, 3, 3]
        out_shape = list(sixd.shape)
        out_shape[-1] = 3
        in_dims = len(sixd.shape)
        view = [-1, 3, 2]
        if in_dims == 2:
            out_shape.append(3)
        else:
            view.insert(1, sixd.shape[-3])
        return roma.special_gramschmidt(sixd.view(*view)).view(out_shape)