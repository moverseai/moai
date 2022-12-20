from moai.supervision.losses.distribution.kl_divergence import KL

import torch
import functools

__all__ = ["Lambda", "JS"]

class Lambda(KL):
    def __init__(self,
        lamda: float=0.5, # default lambda for Jensen-Shannon Divergence
        is_input_log:   bool=False,
        is_target_log:  bool=False,
        epsilon: float=1e-24,
    ):
        super(Lambda, self).__init__(True, True)
        self.lamda = lamda
        self.is_input_log_ = is_input_log
        self.is_target_log_ = is_target_log
        self.epsilon = epsilon

    def forward(self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor=None, # float tensor
    ) -> torch.Tensor:
        m = (
            self.lamda * (pred.exp() if self.is_input_log_ else pred)
            + 
            (1.0 - self.lamda) * (gt.exp() if self.is_target_log_ else gt)
        )
        m = (m + self.epsilon).log() 
        p = pred if self.is_input_log_ else (pred + self.epsilon).log()
        g = gt if self.is_target_log_ else (gt + self.epsilon).log()
        
        pred_to_m = super(Lambda, self).forward(p, m)
        gt_to_m = super(Lambda, self).forward(g, m)
        lamda_divergence = self.lamda * pred_to_m + (1.0 - self.lamda) * gt_to_m
        if weights is not None:
            lamda_divergence = lamda_divergence * weights
        return lamda_divergence

JS = functools.partial(Lambda, lamda=0.5)