import torch

#NOTE: https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/4

__all__ = ["CrossEntropy"]

#TODO: to be refactored into a generic multi-/binary classification loss (/w logits, logs or probs in gt/pred)
class CrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self,
    
    ):
        super(CrossEntropy, self).__init__()

    def forward(self, 
        gt: torch.Tensor, # class ids
        pred: torch.Tensor, # logits (i.e. raw predictions, no softmax applied -- see link above)
    ) -> torch.Tensor:
        return super(CrossEntropy, self).forward(pred, gt)