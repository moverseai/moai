from moai.utils.arguments import ensure_choices
from functools import partial

import torch
import logging

log = logging.getLogger(__name__)

class Classified(torch.nn.Module):
    
    __CHOICES__ = ['eq', 'neq']

    def __init__(self,
        prediction:       str='eq',
        groundtruth:      str='eq',
    ):
        super(Classified, self).__init__()
        ensure_choices(log, 'Classified Prediction Equality', prediction, Classified.__CHOICES__)
        ensure_choices(log, 'Classified Groundtruth Equality', groundtruth, Classified.__CHOICES__)
        self.pred_rel = torch.eq if prediction == 'eq' else torch.ne
        self.gt_rel = torch.eq if groundtruth == 'eq' else torch.ne

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        b, num_classes = pred.shape[:2]
        class_indices = torch.arange(num_classes).to(pred.device)
        classes = torch.argmax(pred, dim=1).view(b, 1, -1)
        gt = gt.view(b, 1, -1)
        correct = self.pred_rel(classes, class_indices) & self.gt_rel(gt, class_indices)
        correct = correct.float()
        if weights is not None:
            correct = correct * weights
        if mask is not None:
            correct = correct[mask]
        return correct.sum(dim=0).squeeze()

TruePositive = partial(Classified, prediction='eq', groundtruth='eq')
TrueNegative = partial(Classified, prediction='neq', groundtruth='neq')
FalsePositive = partial(Classified, prediction='eq', groundtruth='neq')
FalseNegative = partial(Classified, prediction='neq', groundtruth='eq')