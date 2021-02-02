import torch

import functools

__all__ = [
    "VisibilityFOV",
    "VisibilityHeatmap"
]

class VisibilityFOV(torch.nn.Module):
    def __init__(self,
        width:          int=1,
        height:         int=1,
        coord_type:     str='ndc',              # [B, K, UV]
    ):
        super(VisibilityFOV,self).__init__()
        self.coord_type = coord_type
        self.width = width
        self.height = height

    def forward(self,
        coords:   torch.Tensor
    ) -> torch.Tensor:  
        _coords = coords.clone().detach()
        if self.coord_type != 'coord':
            _coords[..., 0] = (_coords[..., 0] + 1.0) / 2.0 * self.width if self.coord_type == 'ndc' else _coords[..., 0] * self.width
            _coords[..., 1] = (_coords[..., 1] + 1.0) / 2.0 * self.height if self.coord_type == 'ndc' else _coords[..., 1] * self.height
        masks = torch.zeros_like(coords)
        masks[..., 0] = (_coords[..., 0] >= 0) * (_coords[..., 0] < self.width)
        masks[..., 1] = (_coords[..., 1] >= 0) * (_coords[..., 1] < self.height)
        return masks

class VisibilityHeatmap(torch.nn.Module):

    def _mask(self,
        coords:     torch.Tensor,           # [B, K, UV]
        heatmaps:   torch.Tensor            # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor: 
        r"""Extracts the visibility mask of keypoint predictions based on heatmap values.  .

        Args:
            coords:     torch.Tensor,               [B, K, UV(S) or (S)VU]
            heatmaps:   torch.Tensor                [B, K, (D), H, W], with its value across the spatial dimensions summing to unity

        Returns:
            The visibility weights of coordinates,  [B, K, UV(S) or (S)VU].
        """
        masks = torch.zeros_like(coords)
        channels = heatmaps.shape[1]
        for i in range(channels):
            heatmap = heatmaps[:, i, ...]
            for b in range(coords.shape[0]):
                uv = tuple(coords.flip(-1).long()[b, i])
                if uv[0] > -1 and uv[1] > -1 and uv[0] < heatmap.shape[-2] and uv[1] < heatmap.shape[-1]:
                    masks[b, i, ...] = 1.0 if heatmap[b][uv] > self.threshold else 0.0
                else:
                    masks[b, i, ...] = 0.0
        return masks

    def __init__(self,
        width:          int=1,
        height:         int=1,
        threshold:      float=0.4,
        coord_type:     str='ndc',              # [B, K, UV]
    ):
        super(VisibilityHeatmap,self).__init__()
        self.coord_type = coord_type
        self.width = width
        self.height = height
        self.threshold = threshold

    def forward(self,
        coords:     torch.Tensor,           # [B, K, UV(S) or (S)VU]
        heatmaps:   torch.Tensor            # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:                      # [B, K]
        _coords = coords.clone().detach()
        if self.coord_type != 'coord':
            _coords[..., 0] = (_coords[..., 0] + 1.0) / 2.0 * self.width if self.coord_type == 'ndc' else _coords[..., 0] * self.width
            _coords[..., 1] = (_coords[..., 1] + 1.0) / 2.0 * self.height if self.coord_type == 'ndc' else _coords[..., 1] * self.height
        masks = self._mask(_coords, heatmaps)
        return masks


