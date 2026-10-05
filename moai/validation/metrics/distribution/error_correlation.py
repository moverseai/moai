import numpy as np
import torch

from moai.validation.metric import MoaiMetric

__all__ = ["ErrorConfidenceCorrelation"]


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Average-tie ranks along the last dim, for Spearman."""
    order = x.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=x.dtype)
    arange = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).expand_as(x)
    ranks.scatter_(-1, order, arange)
    return ranks


class ErrorConfidenceCorrelation(MoaiMetric):
    def __init__(self, mode: str = "pearson", eps: float = 1e-8) -> None:
        super().__init__()
        if mode not in ("pearson", "spearman"):
            raise ValueError(f"mode must be 'pearson' or 'spearman', got {mode!r}")
        self.mode = mode
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,  # [B, K, 2]
        gt: torch.Tensor,  # [B, K, 2]
        sigma: torch.Tensor,  # [B, K], any scalar monotonic with uncertainty
    ) -> torch.Tensor:
        error = (pred - gt).norm(dim=-1).reshape(-1)  # [B*K]
        confidence = sigma.reshape(-1)
        if self.mode == "spearman":
            error = _rank(error.unsqueeze(0)).squeeze(0)
            confidence = _rank(confidence.unsqueeze(0)).squeeze(0)
        error = error - error.mean()
        confidence = confidence - confidence.mean()
        return (error * confidence).sum() / (
            error.norm() * confidence.norm() + self.eps
        )

    def compute(self, correlations: np.ndarray) -> np.ndarray:
        return correlations.mean()
