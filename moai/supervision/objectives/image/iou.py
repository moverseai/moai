import torch

from moai.monads.utils import spatial_dim_list

__all__ = ["IoU"]


class IoU(torch.nn.Module):
    def __init__(
        self,
        reduce: bool = False,  # False when used as a loss, True when used as a metric
    ):
        super().__init__()
        self.reduce = reduce

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        dims = spatial_dim_list(pred)
        intersect = (pred * gt).sum(dims)
        union = (pred + gt - (pred * gt)).sum(dims) + 1e-6
        return 1 - (intersect / union)
        # return (
        #     (
        #         (intersect / union).sum()
        #         / intersect.numel()  # NOTE: is correct for batch size = 1 only
        #     )
        #     if self.reduce
        #     else intersect / union
        # )
