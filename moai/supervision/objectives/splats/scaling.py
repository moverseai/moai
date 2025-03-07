import torch

__all__ = ["LongAndThinPenalty"]


class LongAndThinPenalty(torch.nn.Module):
    def __init__(
        self,
        max_scale: float = 0.008,
        scale_ratio: float = 10.0,
    ):
        super().__init__()
        self.max_scale = max_scale
        self.scale_ratio = scale_ratio

    def forward(self, scaling: torch.Tensor) -> torch.Tensor:
        max_vals = scaling.max(dim=-1).values
        min_vals = scaling.min(dim=-1).values
        ratio = max_vals / min_vals
        thresh_idxs = (max_vals > self.max_scale) & (ratio > self.scale_ratio)
        return (
            max_vals[thresh_idxs].mean()
            if thresh_idxs.sum() > 0
            else torch.scalar_tensor(0.0, device=scaling.device)
        )
