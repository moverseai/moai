import numpy as np

__all__ = ['rmse']

def rmse(rmse: np.ndarray, threshold: float) -> bool:
    return float(rmse) > threshold
    