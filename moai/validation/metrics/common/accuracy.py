from functools import partial

import torch


class Accuracy(torch.nn.Module):
    def __init__(
        self,
        topk: int = 1,
    ):
        super(Accuracy, self).__init__()
        self.topk = topk

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        values, indices = torch.topk(
            pred, k=self.topk, dim=1, largest=True, sorted=True
        )
        b = gt.shape[0]
        indices = indices.view(b, self.topk, -1)
        gt = gt.view(b, 1, -1).expand_as(indices)
        correct = torch.eq(gt, indices)
        correct = correct.sum(dim=1).float()
        if weights is not None:
            correct = correct * weights
        if mask is not None:
            correct = correct[mask]
        return correct.mean()


TopAccuracy = partial(Accuracy, topk=1)
Top5Accuracy = partial(Accuracy, topk=5)
