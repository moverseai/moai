import kornia
import torch

__all__ = ["StructuralDissimilarity"]

# NOTE: check if kornia fixes its implementation https://github.com/kornia/kornia/issues/473
# NOTE: was fixed and adapted

# TODO: adapt to support multiplicative SSIM: https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.pdf
# TODO: other implementations to check:
#       - https://github.com/Jack-guo-xy/Python-IW-SSIM
#       - https://github.com/francois-rozet/piqa
#       - https://github.com/lartpang/mssim.pytorch
#       - https://github.com/VainF/pytorch-msssim


class StructuralDissimilarity(kornia.losses.SSIMLoss):
    def __init__(self, window_size: int = 7, dynamic_range: float = 1.0):
        super().__init__(
            window_size=window_size, reduction="none", max_val=dynamic_range
        )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            assert mask.dtype == torch.bool or mask.dtype == torch.float32
            if mask.dtype == torch.bool:
                gt = torch.where(mask, gt, torch.zeros_like(gt))
                pred = torch.where(mask, pred, torch.zeros_like(gt))
            elif mask.dtype == torch.float32:
                cond = mask > 0.0
                gt = torch.where(cond, gt, torch.zeros_like(gt))
                pred = torch.where(cond, pred, torch.zeros_like(gt))
        dssim = super().forward(gt, pred)
        if weights is not None:
            dssim = dssim * weights
        if mask is not None:
            dssim = dssim[mask]
        # return torch.clamp(1.0 - (ssim + 1.0) * 0.5, min=0.0, max=1.0)
        return dssim
