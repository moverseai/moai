import torch
import kornia

__all__ = ["StructuralDisimilarity"]

#NOTE: check if kornia fixes its implementation https://github.com/kornia/kornia/issues/473 

class StructuralDisimilarity(kornia.losses.SSIM):
    def __init__(self,
        window_size: int=7,        
        dynamic_range: float=1.0
    ):
        super(StructuralDisimilarity, self).__init__(
            window_size=window_size,
            reduction='none',
            max_val=dynamic_range
        )

    def forward(self, 
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor=None,
        mask: torch.Tensor=None,
    ) -> torch.Tensor:                    
        if mask is not None:
            gt = torch.where(mask, gt, torch.zeros_like(gt))
            pred = torch.where(mask, pred, torch.zeros_like(gt))            
        ssim = super(StructuralDisimilarity, self).forward(gt, pred)
        if weights is not None:
            ssim = ssim * weights
        if mask is not None:
            ssim = ssim[mask]
        return torch.clamp(1.0 - (ssim + 1.0) * 0.5, min=0.0, max=1.0)