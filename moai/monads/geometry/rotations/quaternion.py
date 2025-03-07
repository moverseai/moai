import roma
import torch

__all__ = ["QuaternionComposition"]


class QuaternionComposition(torch.nn.Module):
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # assert x.shape[-1] == 4
        # assert y.shape[-1] == 4
        return roma.quat_composition((x, y), normalize=self.normalize)
