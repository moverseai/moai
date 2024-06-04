import functools

import torch

from moai.validation.metrics.object_pose.rotation import _angular_error

__all__ = [
    "Accuracy2",
    "Accuracy5",
    "Accuracy10",
]


def _calculate_acc(
    gt_rot: torch.Tensor,
    pr_rot: torch.Tensor,
    gt_pos: torch.Tensor,
    pr_pos: torch.Tensor,
    threshold: float,
):
    """
    An estimated pose is correct if the average distance
    is smaller than 5pixels.k◦,  k  cm as proposed  in  Shotton  et  al.  (2013).
    The  5◦,5cm metric considers an estimated pose to be correct if
    its rotation error is within 5◦ and the translation error is below 5cm.
    Provide also the results with 2◦, 2cm and 10◦, 10 cm.
    input: calculates the error per batch size
    """
    b, _, __ = gt_rot.shape
    rotation_error = _angular_error(gt_rot, pr_rot, radians=False)
    translation_error = (
        torch.linalg.norm(gt_pos - pr_pos, ord=2, dim=-1) * 100.0
    )  # in cm
    # translation_error = torch.linalg.norm(gt_pos - pr_pos, ord=2, dim=-1) / 10.0 # convert to cm
    # count correct threshold
    condition_thres = torch.where(
        (rotation_error < threshold) & (translation_error < threshold), 1.0, 0.0
    )
    count_thres = torch.count_nonzero(condition_thres)
    acc = 100 * float(count_thres) / b
    return torch.tensor(acc).float()


class Accuracy(torch.nn.Module):
    def __init__(
        self,
        threshold: int = 10,
    ):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(
        self,
        pred_rotation: torch.Tensor,
        gt_rotation: torch.Tensor,
        pred_position: torch.Tensor,
        gt_position: torch.Tensor,
    ) -> torch.Tensor:
        return _calculate_acc(
            gt_rotation, pred_rotation, gt_position, pred_position, self.threshold
        )


Accuracy2 = functools.partial(Accuracy, threshold=2)

Accuracy5 = functools.partial(Accuracy, threshold=5)

Accuracy10 = functools.partial(Accuracy, threshold=10)
