from moai.supervision.losses.regression.L1 import L1
from moai.supervision.losses.regression.L2 import L2
from moai.supervision.losses.regression.cosine_distance import CosineDistance
from moai.supervision.losses.regression.distance import Distance
from functools import partial

LogL1 = partial(L1, mode='log')
LnL1 = partial(L1, mode='ln')

__all__ = [
    "CosineDistance",
    "L1",
    "LogL1",
    "LnL1",
    "L2",
    "Distance",
]