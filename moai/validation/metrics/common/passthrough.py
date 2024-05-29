from moai.validation.metric import MoaiMetric

import torch
import numpy as np

class Passthrough(MoaiMetric):
    def __init__(self):
        super().__init__()

    def forward(self,
        feat:       torch.Tensor,
    ) -> torch.Tensor:
       return feat
    
    def compute(self, feats: np.ndarray) -> np.ndarray:
        return feats