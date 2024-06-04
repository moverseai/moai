import functools

from moai.monads.filter.highpass.finite_difference import FiniteDifference
from moai.monads.filter.highpass.image2d import Diff2d, Laplacian2d, Sobel2d

Backward1d = functools.partial(FiniteDifference, dims=[2], mode="backward")
Forward1d = functools.partial(FiniteDifference, dims=[2], mode="forward")
Central1d = functools.partial(FiniteDifference, dims=[2], mode="central")

Backward2d = functools.partial(FiniteDifference, dims=[2, 3], mode="backward")
Forward2d = functools.partial(FiniteDifference, dims=[2, 3], mode="forward")
Central2d = functools.partial(FiniteDifference, dims=[2, 3], mode="central")

Backward3d = functools.partial(FiniteDifference, dims=[2, 3, 4], mode="backward")
Forward3d = functools.partial(FiniteDifference, dims=[2, 3, 4], mode="forward")
Central3d = functools.partial(FiniteDifference, dims=[2, 3, 4], mode="central")

__all__ = [
    "FiniteDifference",
    "Backward1d",
    "Forward1d",
    "Central1d",
    "Backward2d",
    "Forward2d",
    "Central2d",
    "Backward3d",
    "Forward3d",
    "Central3d",
    "Laplacian2d",
    "Sobel2d",
    "Diff2d",
]
