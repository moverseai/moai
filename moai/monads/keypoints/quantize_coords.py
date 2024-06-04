import functools

import torch

from moai.monads.utils import spatial_dims

__all__ = [
    "QuantizeCoords",
]


class QuantizeCoords(torch.nn.Module):
    def _convert_norm(
        self,
        uvs: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        dims = torch.Tensor([width, height]).to(uvs)
        return uvs / dims

    def _convert_ndc(
        self,
        uvs: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        one = self._convert_one(uvs, height, width)
        return one * 2.0 - 1.0

    def __init__(
        self,
        mode: str = "round",  # round, ceil, floor
        flip: bool = False,
        coord_type: str = "ndc",
        width: int = 1,
        height: int = 1,
    ):
        super(QuantizeCoords, self).__init__()
        self.mode = mode
        self.flip = flip
        self.coord_type = coord_type
        self.width = width
        self.height = height
        if self.mode == "round":
            self.branch = functools.partial(torch.round)
        elif self.mode == "ceil":
            self.branch = functools.partial(torch.ceil)
        elif self.mode == "floor":
            self.branch = functools.partial(torch.floor)
        else:
            raise ValueError("{} mode is not valid.".format(self.mode))

    def forward(
        self,
        coords: torch.Tensor,
        grid: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.coord_type != "coord":
            coords[..., 0] = (
                (coords[..., 0] + 1.0) / 2.0 * self.width
                if self.coord_type == "ndc"
                else coords[..., 0] * self.width
            )
            coords[..., 1] = (
                (coords[..., 1] + 1.0) / 2.0 * self.height
                if self.coord_type == "ndc"
                else coords[..., 1] * self.height
            )
        coords = self.branch(coords)

        if self.coord_type != "coord":
            coords = (
                self._convert_ndc(coords, self.height, self.width)
                if self.coord_type == "ndc"
                else self._convert_norm(coords, self.height, self.width)
            )
        # if grid is not None:
        #     dims = spatial_dims(grid)
        #     for b in range(coords.shape[0]):
        #         for j in range(coords.shape[1]):
        #             for d in range(len(dims)):
        #                 coords[b, j, d] = grid[b, d][tuple(coords.int()[b, j].flip(-1))]
        return coords.flip(-1) if self.flip else coords
