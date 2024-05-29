from moai.validation.metric import MoaiMetric

import torch
import numpy as np

__all__ = ["FID"]

class FID(MoaiMetric):
    def __init__(self):
        super().__init__()

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        # weights:    torch.Tensor=None,
    ) -> torch.Tensor:
        return {
            'gt': gt, 'pred': pred    
        }

    def compute(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        num_samples = pred.shape[0] if pred.shape[0] <= gt.shape[0] else gt.shape[0] 
        mu1, mu2= np.mean(gt[:num_samples], axis=0), np.mean(pred[:num_samples], axis=0)
        c1, c2 = np.cov(gt[:num_samples]), np.cov(pred[:num_samples])
        diff_sq = np.linalg.norm(mu1 - mu2, ord=2, axis=-1, keepdims=False)
        fid = diff_sq + np.trace(c1 + c2 - 2*np.sqrt(c1*c2))
        # if weights is not None:
        #     fid = fid * weights
        return fid