import logging
import typing

import torch
from torchmetrics.classification import MulticlassAccuracy

log = logging.getLogger(__name__)


class MultiClassAcc(MulticlassAccuracy):
    def __init__(
        self,
        num_classes: int,
        # compute_on_step: bool=True,
        # dist_sync_on_step: bool=False,
        # process_group: typing.Any=None,
        # dist_sync_fn: typing.Callable=None,
        # compute: bool=True,
    ):
        super(MulticlassAccuracy, self).__init__(
            num_classes=num_classes,
            # threshold=threshold,
            # compute_on_step=compute_on_step,
            # dist_sync_on_step=dist_sync_on_step,
            # process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
            # compute=compute,
        )
        log.info(f"Initialized Multiclass Accuracy with {num_classes} classes")

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return super(MulticlassAccuracy, self).forward(
            preds=pred,
            target=gt,
        )
