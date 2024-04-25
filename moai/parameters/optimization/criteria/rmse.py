import torch

__all__ = ['rmse']

def rmse(rmse: torch.Tensor, threshold: float) -> bool:
    return float(rmse) > threshold
    