import torch

__all__ = ['Rotate']

class Rotate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
        rotation:   torch.Tensor,
        points:     torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum('bij,bpj->bpi', rotation, points)
