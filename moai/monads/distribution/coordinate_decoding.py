import functools

import torch

from moai.monads.utils import (
    expand_spatial_dims,
    flatten_spatial_dims,
    spatial_dim_list,
    spatial_dims,
)

__all__ = ["CoordinateDecoding"]


# NOTE: generic helper functions should go to utils?
def unravel_indices(indices: torch.Tensor, shape: torch.Tensor) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []
    for i, dim in enumerate(reversed(shape)):
        coord.append(indices % dim)
        indices = indices // dim
    coord = torch.stack(coord[::-1], dim=-1).long()
    return coord


class CoordinateDecoding(torch.nn.Module):

    @staticmethod
    def _argmax(
        grid: torch.Tensor,  # [B, (S)VU, (D), H, W], order is (S)VU not UV(S), y coord first channel
        heatmaps: torch.Tensor,  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:  # [B, K, UV(S) or (S)VU]
        channels = heatmaps.shape[1]
        dims = spatial_dims(heatmaps)
        coords = []
        for i in range(channels):  # TODO: refactor for loop
            heatmap = heatmaps[:, i, ...].unsqueeze(1)
            flattened_heatmap = flatten_spatial_dims(heatmap).squeeze(1)
            indices = torch.argmax(flattened_heatmap, dim=1)
            raw_m = unravel_indices(indices, dims)
            coord = torch.zeros_like(raw_m).float()
            for b in range(coord.shape[0]):
                for j in range(len(dims)):
                    coord[b, j] = grid[b, j][tuple(raw_m[b])]
            coords.append(coord)
        return torch.stack(coords, dim=1)

    @staticmethod
    def _argmax_sub(
        grid: torch.Tensor,  # [B, (S)VU, (D), H, W], order is (S)VU not UV(S), y coord first channel
        heatmaps: torch.Tensor,  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
        sub: float,  # sub-pixel shifting to second maximal activation
    ) -> torch.Tensor:  # [B, K, UV(S) or (S)VU]
        channels = heatmaps.shape[1]
        dims = spatial_dims(heatmaps)
        argmax_dims = spatial_dim_list(heatmaps)
        coords = []
        for i in range(channels):  # TODO: refactor for loop
            heatmap = heatmaps[:, i, ...].unsqueeze(1)
            flattened_heatmap = flatten_spatial_dims(heatmap).squeeze(1)
            indices = torch.topk(flattened_heatmap, 2, dim=1)[1]

            raw_m = unravel_indices(indices[:, 0], dims)
            m = torch.zeros_like(raw_m).float()
            for b in range(m.shape[0]):
                for j in range(len(dims)):
                    m[b, j] = grid[b, j][tuple(raw_m[b])]

            raw_s = unravel_indices(indices[:, 1], dims)
            s = torch.zeros_like(raw_s).float()
            for b in range(s.shape[0]):
                for j in range(len(dims)):
                    s[b, j] = grid[b, j][tuple(raw_s[b])]

            coord = m + sub * (s - m) * (
                grid.amax(dim=argmax_dims) - grid.amin(dim=argmax_dims)
            ) / (torch.linalg.norm(s - m) * dims)
            coords.append(coord)
        return torch.stack(coords, dim=1)

    def __init__(
        self,
        mode: str = "argmax",  # NOTE: to support argmax, standard
        flip: bool = True,  # flip order of coordinates as originated from the grid's order
        sub: float = 0.25,
    ):
        super(CoordinateDecoding, self).__init__()
        self.mode = mode
        self.flip = flip
        self.sub = sub
        self.branch = (
            functools.partial(self._argmax_sub, sub=sub)
            if mode == "standard"
            else functools.partial(self._argmax)
        )

    def forward(
        self,
        grid: torch.Tensor,  # coordinates grid tensor of C coordinates
        heatmaps: torch.Tensor,  # spatial probability tensor of K keypoints
    ) -> torch.Tensor:
        coords = self.branch(grid, heatmaps)
        return torch.flip(coords, dims=[2]) if self.flip else coords
