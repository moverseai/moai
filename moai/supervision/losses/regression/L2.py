import torch

class L2(torch.nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor=None, # float tensor
        mask: torch.Tensor=None, # byte tensor
    ) -> torch.Tensor:
        l2 = ((gt - pred) ** 2)
        if weights is not None:
            l2 = l2 * weights
        if mask is not None:
            l2 = l2[mask]        
        return l2