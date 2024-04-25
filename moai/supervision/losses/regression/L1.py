from moai.utils.arguments import ensure_choices

import torch
import logging

log = logging.getLogger(__name__)

__all__ = ['L1']

class L1(torch.nn.Module):
    
    __MODES__ = ['raw', 'ln', 'log']
    __MODES_MAP__ = {
        'raw': lambda e: e,
        'ln': torch.log,
        'log': torch.log10,
    }

    def __init__(self,
        mode:       str='raw', # one of ['raw', 'ln', 'log']
        bias:       float=0.0,
    ):
        super(L1, self).__init__()
        mode = ensure_choices(log, 'mode', mode, L1.__MODES__)
        self.mode = L1.__MODES_MAP__[mode]
        self.bias = bias

    def forward(self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor=None, # float tensor
        mask: torch.Tensor=None, # byte tensor
    ) -> torch.Tensor:
        l1 = torch.abs(gt - pred) 
        if weights is not None:
            l1 = l1 * weights
        if mask is not None:
            l1 = l1[mask]        
        return self.mode(l1 + self.bias)