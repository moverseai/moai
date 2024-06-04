import functools

import torch


def _calculate_projection_acc(
    projected_gt: torch.Tensor, projected_est: torch.Tensor, threshold: float
):
    """
    We accept  a  pose estimation as correct when the 2D projection error is smaller than
    a predefined threshold.
    Input:
    predicted pxls based on the estimated pose and gt pixels derived from gt pose.
    """
    b, _, __ = projected_gt.shape
    e = torch.linalg.norm(projected_est - projected_gt, ord=2, dim=-1).mean(dim=-1)
    valid_invalid = torch.where(e < threshold, 1, 0)
    valid = torch.count_nonzero(valid_invalid)
    acc = 100.0 * float(valid) / b
    return torch.tensor(acc).float()


class Projection(torch.nn.Module):
    def __init__(
        self,
        threshold: int = 5,
    ):
        super(Projection, self).__init__()
        self.threshold = threshold

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        return _calculate_projection_acc(gt, pred, self.threshold)


Projection2 = functools.partial(Projection, threshold=2)

Projection5 = functools.partial(Projection, threshold=5)

Projection10 = functools.partial(Projection, threshold=10)
