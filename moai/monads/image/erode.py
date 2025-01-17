import kornia
import torch

__alL__ = ["Erosion"]


class Erosion(torch.nn.Module):
    def __init__(
        self,
        pixels: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer("kernel", torch.ones(1 + 2 * pixels, 1 + 2 * pixels))

    def forward(
        self,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return kornia.morphology.erosion(mask, self.kernel)
