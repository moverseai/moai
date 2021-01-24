import torch

__all__ = ["NormalizedPositionError"]

def _normalised_position_error(
    gt: torch.Tensor,
    pred: torch.Tensor,
):
    l2_norm = torch.linalg.norm(gt - pred, ord=2, dim=-1)
    return l2_norm / (torch.linalg.norm(gt, ord=2, dim=-1) + 1e-7)

#TODO: extract eps to param
class NormalizedPositionError(torch.nn.Module):
    def __init__(self,

    ):
        super(NormalizedPositionError, self).__init__()

    def forward(self,     
        gt: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        return _normalised_position_error(gt, pred).mean()