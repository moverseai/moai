import kornia
import torch

__all__ = ["EDT"]


# NOTE: distance towards non-zero pixel
class EDT(kornia.contrib.DistanceTransform):
    def __init__(
        self,
        kernel_size: int = 3,
        softmin_factor: float = 0.35,
        full: bool = False,
    ):
        super().__init__(kernel_size=kernel_size, h=softmin_factor)
        self.full = full

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # scale = np.sqrt(np.prod(np.array([*mask.shape[2:]]) ** 2))
        scale = 1.0
        inverse = 1.0 - mask
        edt = super().forward(image=mask)
        return scale * (
            edt if not self.full else inverse * edt + super().forward(image=inverse)
        )
