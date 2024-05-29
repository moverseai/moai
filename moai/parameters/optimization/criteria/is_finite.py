import numpy as np

__all__ = ['is_finite']

def is_finite(monitor: np.ndarray) -> bool:
    return np.any(np.logical_not(np.isfinite(monitor)))
    