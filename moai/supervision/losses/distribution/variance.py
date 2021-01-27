from moai.monads.utils import expand_spatial_dims
from moai.monads.distribution import CenterOfMass

import torch

__all__ = ["VarianceRegularization"]

__KEYPOINTS_TO_COORDS__ = {#TODO: create grid conversion OPs and link to there
    'ndc': lambda coord, img: torch.addcmul(
        torch.scalar_tensor(0.5).to(coord), coord, torch.scalar_tensor(0.5).to(coord)
    ) * torch.Tensor([*img.shape[2:]]).to(coord).expand_as(coord),
    'coord': lambda coord, img: coord,
    'norm': lambda coord, img: coord * torch.Tensor([*img.shape[2:]]).to(coord).expand_as(coord),
}

__GRID_TO_COORDS__ = {#TODO: create grid conversion OPs and link to there
    'ndc': lambda grid: torch.addcmul(
        torch.scalar_tensor(0.5).to(grid), grid, torch.scalar_tensor(0.5).to(grid)
    ) * expand_spatial_dims(torch.Tensor([*grid.shape[2:]]).to(grid), grid),
    'coord': lambda grid: grid,
    'norm': lambda grid: grid * expand_spatial_dims(torch.Tensor([*grid.shape[2:]]).to(grid), grid),
}

class VarianceRegularization(CenterOfMass):
    def __init__(self,
        sigma: float=1.5, # in pixels
        grid_type: str='ndc', # 'ndc', 'coord', 'norm'        
    ):
        super(VarianceRegularization, self).__init__(mode='coords', flip=False)
        self.target_variance = sigma ** 2
        self.grid_type = grid_type

    def forward(self, 
        heatmaps:   torch.Tensor,        # [B, K, (D), (H), W]
        grid:       torch.Tensor,        # [B, (1-3), (D), (H), W]
        keypoints:  torch.Tensor,        # [B, K, (1-3)]
    ) -> torch.Tensor:        
        k = __KEYPOINTS_TO_COORDS__[self.grid_type](keypoints)        
        grid = __GRID_TO_COORDS__[self.grid_type](grid)
        diffs = (grid.unsqueeze(1) - k) ** 2 # [B, K, (1-3), (D), (H), W]
        pred_stds = super(VarianceRegularization, self).forward(diffs, heatmaps)
        pred_variances = pred_stds ** 2
        squared_error = (pred_variances - self.target_variance) ** 2
        return torch.sum(squared_error, dim=-1)


