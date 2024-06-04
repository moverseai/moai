import torch

from moai.monads.utils.common import dim_list


class AbsRel(torch.nn.Module):
    def __init__(self):
        super(AbsRel, self).__init__()

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        absrel = torch.abs((gt - pred) / gt)
        if weights is not None:
            absrel = absrel * weights
        if mask is not None:
            absrel = absrel[mask]
        if weights is None:
            return torch.mean(torch.mean(absrel, dim=dim_list(gt)))
        else:
            return torch.mean(
                torch.sum(absrel, dim=dim_list(gt))
                / torch.sum(weights, dim=dim_list(gt))
            )
