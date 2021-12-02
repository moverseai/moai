import torch

#NOTE: https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/4

__all__ = ["CrossEntropy"]

#TODO: to add class weightings
class CrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self,
        ignore_index: int=-100,        
    ):
        super(CrossEntropy, self).__init__(            
            ignore_index=ignore_index,
            reduction='none'
        )

    def forward(self, 
        gt: torch.Tensor, # class ids [B, 1, ...]
        pred: torch.Tensor, # logits [B, C, ...] (i.e. raw predictions, no softmax applied -- see link above)
    ) -> torch.Tensor:
        return super(CrossEntropy, self).forward(pred, gt)