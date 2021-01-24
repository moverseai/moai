import torch

import functools

__all__ = [
    "VisibilityFOV",
    "VisibilityHeatmap"
]

class VisibilityFOV(torch.nn.Module):
    def __init__(self,
        coord_type:     str='ndc',              # [B, K, UV]
        width:          int=1,
        height:         int=1,
    ):
        super(VisibilityFOV,self).__init__()
        self.coord_type = coord_type
        self.width = width
        self.height = height

    def forward(self,
        coords:   torch.Tensor
    ) -> torch.Tensor:  
        if self.coord_type != 'coord':
            coords[..., 0] = (coords[..., 0] + 1.0) / 2.0 * self.width if self.coord_type == 'ndc' else coords[..., 0] * self.width
            coords[..., 1] = (coords[..., 1] + 1.0) / 2.0 * self.height if self.coord_type == 'ndc' else coords[..., 1] * self.height
        masks = torch.zeros(coords.shape[0], coords.shape[1]).to(coords)
        masks[...] = (coords[..., 0] >= 0) * (coords[..., 1] >= 0) * (coords[..., 0] < self.width) * (coords[..., 1] < self.width)
        return masks.bool()

class VisibilityHeatmap(torch.nn.Module):

    def _mask(self,
        coords:     torch.Tensor,           # [B, K, UV]
        heatmaps:   torch.Tensor            # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor: 
        r"""Extracts the visibility mask of keypoint predictions based on heatmap values.  .

        Args:
            coords:     torch.Tensor,           [B, K, UV(S) or (S)VU]
            heatmaps:   torch.Tensor            [B, K, (D), H, W], with its value across the spatial dimensions summing to unity

        Returns:
            The visibility mask of coordinates, [B, K].
        """
        masks = torch.zeros(coords.shape[0], coords.shape[1]).to(coords)
        channels = heatmaps.shape[1]        
        for i in range(channels):
            heatmap = heatmaps[:, i, ...]
            for b in range(coords.shape[0]):
                uv = tuple(coords.flip(-1).long()[b, i])
                if uv[0] > -1 and uv[1] > -1 and uv[0] < heatmap.shape[-2] and uv[1] < heatmap.shape[-1]:
                    masks[b, i] = 1.0 if heatmap[b][uv] > self.threshold else 0.0
                else:
                    masks[b, i] = 0.0
        return masks.bool()

    def __init__(self,
        threshold:      float=0.4
    ):
        super(VisibilityHeatmap,self).__init__()
        self.threshold = threshold

    def forward(self,
        coords:     torch.Tensor,           # [B, K, UV(S) or (S)VU]
        heatmaps:   torch.Tensor            # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:                      # [B, K]
        masks = self._mask(coords, heatmaps)
        return masks


