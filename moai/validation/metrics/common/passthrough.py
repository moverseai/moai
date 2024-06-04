import numpy as np
import torch

from moai.validation.metric import MoaiMetric


class Passthrough(MoaiMetric):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        return feat

    def compute(self, feats: np.ndarray) -> np.ndarray:
        return feats
