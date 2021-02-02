import torch
import functools

__all__ = [
    "PCKh2D_50",
    "PCKh2D_10"
]

def _calculate_pck(
    gt_kpts:        torch.Tensor,
    pred_kpts:      torch.Tensor,
    gt_masks:       torch.Tensor,
    pred_masks:     torch.Tensor,
    kpts_indices:   list,
    threshold:      float
):
    """
    Detected joint is considered correct if the distance between 
    the predicted and the ground truth joint is within a certain threshold (threshold varies)
    """
    diff = torch.abs(gt_kpts - pred_kpts)
    max_sizes = torch.abs(gt_kpts[:, kpts_indices[0], ...] - gt_kpts[:, kpts_indices[1], ...]).unsqueeze(1)    
    kpts_error_norm = diff / max_sizes
    #count correct threshold
    valid_kpts = torch.zeros_like(kpts_error_norm)
    valid_kpts[kpts_error_norm < threshold] = 1.0
    values2compare = torch.logical_or(pred_masks, gt_masks)
    values2penalize = torch.logical_xor(pred_masks, gt_masks)
    valid_kpts[values2penalize] = 0.0
    pck = 100.0 * torch.mean(valid_kpts[values2compare])
    return pck

class PCK(torch.nn.Module):
    def __init__(self,
        threshold:      float=0.05,
        joint_indices:  list=[0, 2]
    ):
        super(PCK, self).__init__()
        self.threshold = threshold
        self.joint_indices = joint_indices

    def forward(self,
        gt_kpts:            torch.Tensor,
        pred_kpts:          torch.Tensor,
        gt_masks:           torch.Tensor,
        pred_masks:         torch.Tensor,
    ) -> torch.Tensor:
        return _calculate_pck(
            gt_kpts, 
            pred_kpts,
            gt_masks, 
            pred_masks, 
            self.joint_indices, 
            self.threshold
        )

PCKh2D_50 = functools.partial(
    PCK, 
    threshold=0.5,
    joint_indices=[2, 3]
)

PCKh2D_10 = functools.partial(
    PCK, 
    threshold=0.1,
    joint_indices=[2, 3]
)
