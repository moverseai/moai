import functools

import torch

__all__ = [
    "SixD2",
    "SixD5",
    "SixD10",
]


def _calculate_add(
    pts_gt: torch.Tensor, pts_est: torch.Tensor, threshold: float, diagonal: float
):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    :pts_est: transformed pts by employing the predicted pose
    :pts_gt: transformed pts by employing the gt pose
    :return: Error of pose_est w.r.t. pose_gt.
    Then  a  pose estimation is considered to be correct if the computed average distance is within threshold% of the model diagonal.
    """
    b, _, __ = pts_gt.size()
    # TODO: check the xyz position if is channel or row ?
    e = torch.linalg.norm(pts_est - pts_gt, ord=2, dim=1).mean(dim=-1)
    threshold /= 100
    valid_invalid = torch.where(e < (threshold * diagonal), 1.0, 0.0)
    valid = torch.count_nonzero(valid_invalid)
    acc = 100.0 * float(valid) / b
    return torch.tensor(acc).float()


class SixD(torch.nn.Module):
    def __init__(
        self,
        diagonal: float,
        threshold: int = 10,
    ):
        super(SixD, self).__init__()
        self.threshold = threshold
        self.diagonal = diagonal

    def forward(
        self,
        pts_gt: torch.Tensor,
        pts_est: torch.Tensor,
        diagonal: torch.Tensor = None,
    ) -> torch.Tensor:
        return _calculate_add(
            pts_gt,
            pts_est,
            self.threshold,
            self.diagonal if diagonal is None else diagonal,
        )


SixD2 = functools.partial(SixD, threshold=2)

SixD5 = functools.partial(SixD, threshold=5)

SixD10 = functools.partial(SixD, threshold=10)
