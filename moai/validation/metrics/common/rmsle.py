import torch

from moai.monads.utils.common import dim_list


class RMSLE(torch.nn.Module):
    def __init__(self, base: str = "natural"):  # 'natural' or 'ten'
        super(RMSLE, self).__init__()
        self.log = torch.log if base == "natural" else torch.log10

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        pred_fix = torch.where(pred == 0.0, pred + 1e-24, pred)
        log_diff_sq = (self.log(gt) - self.log(pred_fix)) ** 2
        if weights is not None:
            log_diff_sq = log_diff_sq * weights
        if mask is not None:
            log_diff_sq = log_diff_sq[mask]
        if weights is None:
            return torch.mean(torch.sqrt(torch.mean(log_diff_sq, dim=dim_list(gt))))
        else:
            log_diff_sq_sum = torch.sum(log_diff_sq, dim=dim_list(gt))
            log_diff_w_sum = torch.sum(weights, dim=dim_list(gt))  # + 1e-18
            return torch.mean(torch.sqrt(log_diff_sq_sum / log_diff_w_sum))
