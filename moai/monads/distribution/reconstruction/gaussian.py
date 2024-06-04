import math

import torch

from moai.monads.utils import expand_spatial_dims, spatial_dim_list

__all__ = ["IsotropicGaussian"]


# TODO: for smaller std see https://dsp.stackexchange.com/questions/23460/how-to-calculate-gaussian-kernel-for-a-small-support-size
class IsotropicGaussian(torch.nn.Module):
    __C__ = math.sqrt(math.pi * 2.0)
    __GRID_TYPE_CHOICES__ = ["ndc", "coord", "norm"]

    def __init__(
        self,
        std: float = 1.5,  # in % of pixels diagonal, relative to grid => can use varying grid inputs
        normalize: bool = True,
        scale: bool = False,
        grid_type: str = "ndc",  # 'ndc', 'coord', 'norm'
        eps: float = 1e-8,
    ):
        super(IsotropicGaussian, self).__init__()
        self.grid_type = grid_type
        std = std / 100.0 * (2.0 if self.grid_type == "ndc" else 1.0)
        self.register_buffer("std", torch.scalar_tensor(std))
        self.normalize = normalize
        self.scale = scale
        self.eps = eps

    def forward(
        self,
        keypoints: torch.Tensor,  # [B, K, (S)UV or UV(S)] with K the number of keypoints
        grid: torch.Tensor,  # [B, (S)UV or UV(S), (D), H, W]
        stds: torch.Tensor = None,  # [B, K]
    ) -> torch.Tensor:  # [B, K, (D), H, W]
        grid_size = torch.Tensor([*grid.shape[2:]])
        scale = 2.0 if self.grid_type == "ndc" else 1.0
        std = scale * (stds if stds is not None else self.std)
        if self.grid_type == "coord":
            std = std * float(torch.Tensor(grid_size).norm(p=2))
        inv_denom = -0.5 * torch.reciprocal(std**2)
        centroids = expand_spatial_dims(keypoints, grid)
        diffs = grid.unsqueeze(1) - centroids
        dist = diffs**2
        nom = torch.sum(dist, dim=2)
        gaussian = torch.exp(nom * inv_denom)
        if self.scale:  # generate a properly scaled Gaussian
            scaling_factor = torch.reciprocal(
                (IsotropicGaussian.__C__ * std) ** len(grid_size)
            )
            scaling_factor = expand_spatial_dims(scaling_factor, grid)
            gaussian = gaussian * scaling_factor
        if self.normalize:  # generate a normalized Gaussian summing to unity
            norm_dims = spatial_dim_list(grid)
            gaussian = gaussian / (
                torch.sum(gaussian, dim=norm_dims, keepdim=True) + self.eps
            )
        return gaussian
