from moai.monads.utils.common import dim_list

import torch

class SqRel(torch.nn.Module):
    def __init__(self):
        super(SqRel, self).__init__()

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        sqrel = ((gt - pred) ** 2) / gt
        if weights is not None:
            sqrel = sqrel * weights
        if mask is not None:
            sqrel = sqrel[mask]
        if weights is None:
            return torch.mean(torch.mean(sqrel, dim=dim_list(gt)))
        else:
            return torch.mean(
                torch.sum(sqrel, dim=dim_list(gt))
                / torch.sum(weights, dim=dim_list(gt))
            )