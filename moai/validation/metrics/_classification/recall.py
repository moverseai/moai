from torchmetrics.classification import MulticlassRecall, MulticlassPrecision


import torch
import typing
import logging

log = logging.getLogger(__name__)


class MultiClassRecall(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        # compute_on_step: bool=True,
        # dist_sync_on_step: bool=False,
        # process_group: typing.Any=None,
        # dist_sync_fn: typing.Callable=None,
        # compute: bool=True,
    ):
        super().__init__(
            # num_classes=num_classes,
            # threshold=threshold,
            # compute_on_step=compute_on_step,
            # dist_sync_on_step=dist_sync_on_step,
            # process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
            # compute=compute,
        )
        self.recall = MulticlassRecall(num_classes=num_classes)
        log.info(f"Initialized Recall with {num_classes} classes")

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # return super(MulticlassPrecision, self).forward(
        #     preds=pred,
        #     target=gt,
        # )
        return self.recall(
            preds=pred,
            target=gt,
        )
