import functools

import torch

__all__ = ["Taylor"]


class Taylor(torch.nn.Module):
    def __init__(
        self,
        grid_type: str = "ndc",  # 'ndc', 'coord', 'norm'
        flip: bool = True,  # flip order of coordinates as originated from the grid's order
    ):
        super(Taylor, self).__init__()
        self.flip = flip
        self.grid_type = grid_type

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

    def _taylor_coord(
        self,
        coord: torch.Tensor,  # [UV]
        hm: torch.Tensor,  # [H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:  # [UV]
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (
                hm[py + 1][px + 1]
                - hm[py - 1][px + 1]
                - hm[py + 1][px - 1]
                + hm[py - 1][px - 1]
            )
            dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])

            derivative = torch.tensor([[dx], [dy]]).to(coord)
            hessian = torch.tensor([[dxx, dxy], [dxy, dyy]]).to(coord)
            if dxx * dyy - dxy**2 != 0:
                hessianinv = torch.inverse(hessian)
                offset = -hessianinv @ derivative
                offset = torch.squeeze(torch.transpose(offset, 0, 1), dim=0)
                coord += offset

        return coord

    def _taylor(
        self,
        coords: torch.Tensor,  # [B, K, UV(S) or (S)VU]
        heatmaps: torch.Tensor,  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:  # [B, K, UV(S) or (S)VU]
        channels = heatmaps.shape[1]
        heatmaps[heatmaps == 0] = 1e-10
        heatmaps = torch.log(heatmaps)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self._taylor_coord(coords[n, p], heatmaps[n, p])
        return coords

    def forward(
        self,
        coords: torch.Tensor,  # coordinates grid tensor of C coordinates
        heatmaps: torch.Tensor,  # spatial probability tensor of K keypoints
    ) -> torch.Tensor:
        _heatmaps = (
            heatmaps.clone().detach()
        )  # TODO: check if clone is needed or if detach suffices
        _coords = coords.clone().detach()
        heatmap_height = _heatmaps.shape[-2]
        heatmap_width = _heatmaps.shape[-1]
        if self.grid_type != "coord":
            _coords[..., 0] = (
                (_coords[..., 0] + 1.0) / 2.0 * heatmap_width
                if self.grid_type == "ndc"
                else _coords[..., 0] * heatmap_width
            )
            _coords[..., 1] = (
                (_coords[..., 1] + 1.0) / 2.0 * heatmap_height
                if self.grid_type == "ndc"
                else _coords[..., 1] * heatmap_height
            )
        _coords = self._taylor(_coords, _heatmaps)
        if self.grid_type == "ndc":
            _coords = self._convert_ndc(_coords, heatmap_height, heatmap_width)
        if self.grid_type == "norm":
            _coords = self._convert_norm(_coords, heatmap_height, heatmap_width)
        return torch.flip(_coords, dims=[-1]) if self.flip else _coords
