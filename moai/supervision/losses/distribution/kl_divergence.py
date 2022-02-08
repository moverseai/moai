import torch
import functools

__all__ = ["KL"]

class KL(torch.nn.KLDivLoss):
    def __init__(self,
        is_input_log:   bool=False,
        is_target_log:  bool=False,
    ):
        super(KL, self).__init__(reduction='none', log_target=is_target_log)
        self.is_input_log = is_input_log

    def forward(self, 
        pred: torch.Tensor,
        gt: torch.Tensor,        
    ) -> torch.Tensor:
        #NOTE: is eps numerical stabilization required?
        return super(KL, self).forward(pred if self.is_input_log else pred.log(), gt)