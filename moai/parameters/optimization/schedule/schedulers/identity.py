import torch

__all__ = ["Identity"]


class Identity(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, **kwargs):
        unity = lambda epoch: 1
        super(Identity, self).__init__(optimizer, unity)

    def __str__(self):
        return "Identity LR scheduler (i.e. no scheduling/LR changes)"
