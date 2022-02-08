import torch
import numpy as np

__all__ = ["CosineDistance"]

#NOTE: see https://github.com/pytorch/pytorch/issues/8069
#TODO: update acos_safe once PR mentioned in above link is merged and available

def _acos_safe(x: torch.Tensor, eps: float=1e-4):
    slope = np.arccos(1.0 - eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = torch.empty_like(x)
    good = torch.abs(x) <= 1.0 - eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1.0 - eps)) - slope * sign * (torch.abs(x[bad]) - 1.0 + eps)
    return buf

'''
    def _acos_safe(x: torch.Tensor, eps: float=1e-4):
        sign = torch.sign(x)
        slope = np.arccos(1.0 - eps) / eps
        return torch.where(torch.abs(x) <= 1.0 - eps,
            torch.acos(x),
            torch.acos(sign * (1.0 - eps)) - slope * sign * (torch.abs(x) - 1.0 + eps)
        )
'''

class CosineDistance(torch.nn.CosineSimilarity):
    def __init__(self,
        dim:                    int=1,
        epsilon:                float=1e-4,
        normalized:             bool=True,
    ):
        super(CosineDistance, self).__init__(dim=dim, eps=epsilon)
        self.normalized = normalized
        self.epsilon = epsilon

    def forward(self,
        pred: torch.Tensor,
        gt: torch.Tensor,        
        weights: torch.Tensor=None, # float tensor
        mask: torch.Tensor=None, # byte tensor
    ) -> torch.Tensor:        
        dot = torch.sum(gt * pred, dim=self.dim) if self.normalized\
            else super(CosineDistance, self).forward(pred, gt)
        # return torch.acos(dot) / np.pi #NOTE: (eps) clamping should also fix the nan grad issue (traditional [-1, 1] clamping does not)
        # return torch.acos(torch.clamp(dot, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)) / np.pi
        return _acos_safe(dot, eps=self.epsilon) / np.pi        
