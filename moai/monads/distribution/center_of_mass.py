import torch

from moai.monads.utils import spatial_dim_list

__all__ = ["CenterOfMass"]


class CenterOfMass(torch.nn.Module):
    def __init__(
        self,  # TODO: to support MoM, DCoM
        mode: str = "default",  # one of ['default', 'default_optimized_2d']
        flip: bool = True,  # flip order of coordinates as originated from the grid's order
    ):
        super(CenterOfMass, self).__init__()
        self.mode = mode
        self.flip = flip

    def _extract_center_of_mass(
        self,
        grid: torch.Tensor,  # [B, (S)VU, (D), H, W], order is (S)VU not UV(S), y coord first channel
        heatmaps: torch.Tensor,  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:  # [B, K, UV(S) or (S)VU]
        channels = heatmaps.shape[1]
        sum_dims = spatial_dim_list(heatmaps)
        CoMs = []
        for i in range(channels):  # TODO: refactor for loop
            heatmap = heatmaps[:, i, ...].unsqueeze(1)
            CoM = torch.sum(heatmap * grid, dim=sum_dims)
            CoMs.append(CoM)
        return torch.stack(CoMs, dim=1)

    def _extract_center_of_mass_optimized2d(
        self,
        grid: torch.Tensor,  # [B, (S)VU, (D), H, W], order is (S)VU not UV(S), y coord first channel
        heatmaps: torch.Tensor,  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:  # [B, K, UV(S) or (S)VU]
        return torch.einsum("bjhw,bchw->bjc", heatmaps, grid)

    def forward(
        self,
        grid: torch.Tensor,  # coordinates grid tensor of C coordinates
        heatmaps: torch.Tensor,  # spatial probability tensor of K keypoints
    ) -> torch.Tensor:
        if self.mode == "default_optimized_2d":
            coms = self._extract_center_of_mass_optimized2d(grid, heatmaps)
        else:
            coms = self._extract_center_of_mass(grid, heatmaps)
        return torch.flip(coms, dims=[2]) if self.flip else coms
