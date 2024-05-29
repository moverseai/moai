from moai.validation.metric import MoaiMetric

import torch
import numpy as np

# Paper: https://arxiv.org/pdf/2007.15240.pdf

__all__ = ["Diversity"]

class Diversity(MoaiMetric):
    def __init__(self):
        super().__init__()

    def forward(self,
        pred:       torch.Tensor,
        # weights:    torch.Tensor=None,
    ) -> torch.Tensor:
        return {
            'pred': pred
        }

    def compute(self, pred: np.ndarray) -> np.ndarray:
        b = int(pred.shape[0] / 2)
        v1 = pred[:b]
        v2 = pred[b:2*b]
        diversity = np.mean(np.linalg.norm(v1 - v2, ord=2, axis=-1, keepdims=False))
        # if weights is not None:
        #     diversity = diversity * weights
        return diversity