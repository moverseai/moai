from moai.monads.math import Abs  # isort:skip
from moai.nn.activation.snake import Snake  # isort:skip
from moai.nn.activation.torch import (  # isort:skip
    BN2d,
    BN2d_LReLu,
    BN2d_ReLu,
    BN2d_ReLu_Drop2d,
    IN2d,
    IN2d_ReLu,
    LReLu_BN,
    LReLu_BN2d,
    LReLu_BN2d_Drop2d,
    LReLu_BN_Drop,
    LReLu_Drop,
    Normalize,
    ReLu_BN,
    ReLu_BN2d,
    ReLu_BN2d_Drop2d,
    ReLu_IN,
    ReLu_IN2d,
)

import logging

import torch

import moai.nn.utils as miu

log = logging.getLogger(__name__)

__all__ = [
    "make_activation",
]

__ACTIVATION_FACTORY__ = {
    "bn2d": BN2d,
    "in2d": IN2d,
    "relu_bn": ReLu_BN,
    "relu_in": ReLu_IN,
    "bn2d_relu": BN2d_ReLu,
    "in2d_relu": IN2d_ReLu,
    "bn2d_relu_drop2d": BN2d_ReLu_Drop2d,
    "relu_bn2d": ReLu_BN2d,
    "relu_in2d": ReLu_IN2d,
    "relu_bn2d_drop2d": ReLu_BN2d_Drop2d,
    "lrelu_bn": LReLu_BN,
    "lrelu_bn_drop": LReLu_BN_Drop,
    "bn2d_lrelu": BN2d_LReLu,
    "lrelu_drop": LReLu_Drop,
    "lrelu_bn2d": LReLu_BN2d,
    "lrelu_bn2d_drop2d": LReLu_BN2d_Drop2d,
    "elu": torch.nn.ELU,
    "relu": torch.nn.ReLU,
    "lrelu": torch.nn.LeakyReLU,
    "abs": Abs,
    "none": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "snake": Snake,
    "normalize": Normalize,
}


def make_activation(
    activation_type: str, features: int, inplace: bool, **kwargs
) -> torch.nn.Module:
    if activation_type in __ACTIVATION_FACTORY__.keys():
        return miu.instantiate(
            __ACTIVATION_FACTORY__[activation_type], **{**locals(), **kwargs}
        )
    else:
        log.error(f"Activation type ({activation_type}) not found.")
