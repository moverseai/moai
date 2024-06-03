from moai.monads.utils.common import dim_list

import torch
import functools

class Delta(torch.nn.Module):
    def __init__(self, 
        threshold: float=1.25
    ):
        super(Delta, self).__init__()
        self.threshold = threshold

    def forward(self,
        gt:             torch.Tensor,
        pred:           torch.Tensor,
        weights:        torch.Tensor=None,
    ) -> torch.Tensor: #NOTE: no mean
        errors =  (torch.max((gt / pred), (pred / gt)) < self.threshold).float()
        if weights is None:
            return torch.mean(errors)
        else:
            return torch.mean(
                torch.sum(errors * weights, dim=dim_list(gt))
                / torch.sum(weights, dim=dim_list(gt))
            )

Delta1 = functools.partial(Delta, threshold=1.25)
Delta2 = functools.partial(Delta, threshold=1.25 ** 2)
Delta3 = functools.partial(Delta, threshold=1.25 ** 3)