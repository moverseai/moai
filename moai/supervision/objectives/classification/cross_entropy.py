import torch

# NOTE: https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/4

__all__ = ["CrossEntropy", "BinaryCrossEntropy"]


# TODO: to add class weightings
class CrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        ignore_index: int = -100,
    ):
        super(CrossEntropy, self).__init__(ignore_index=ignore_index, reduction="none")

    def forward(
        self,
        pred: torch.Tensor,  # logits [B, C, ...] (i.e. raw predictions, no softmax applied -- see link above)
        gt: torch.Tensor,  # class ids [B, 1, ...]
    ) -> torch.Tensor:
        return super(CrossEntropy, self).forward(pred, gt)


class BinaryCrossEntropy(torch.nn.BCELoss):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__(reduction="none")

    def forward(
        self,
        pred: torch.Tensor,  # logits [B, 1, ...] (i.e. raw predictions, no softmax applied -- see link above)
        gt: torch.Tensor,  # class ids [B, 1, ...]
    ) -> torch.Tensor:
        return super(BinaryCrossEntropy, self).forward(pred, gt)
