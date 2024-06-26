import functools
import logging

import torch

import moai.nn.activation as mia
import moai.nn.linear as miln

log = logging.getLogger(__name__)

__all__ = [
    "LinearBlock",
]


class LinearBlock(torch.nn.Module):
    def __init__(
        self,
        linear_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        linear_params: dict,
        activation_params: dict,
        **kwargs: dict
    ):
        super(LinearBlock, self).__init__()
        self.linear = miln.make_linear_op(
            linear_type=linear_type,
            in_features=in_features,
            out_features=out_features,
            **linear_params  # TODO: either merge kwargs here or in the factory method
        )
        self.activation = mia.make_activation(
            features=out_features, activation_type=activation_type, **activation_params
        )
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))
