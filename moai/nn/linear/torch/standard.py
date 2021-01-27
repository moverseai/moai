import moai.nn.activation as mia
import moai.nn.linear as miln

import torch
import functools
import logging

log = logging.getLogger(__name__)

__all__ = [
    "LinearBlock",
]

class LinearBlock(torch.nn.Module):
    def __init__(self,
        linear_type:str,
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
            **linear_params #TODO: either merge kwargs here or in the factory method
        )
        self.activation = mia.make_activation(
            features=out_features,
            activation_type=activation_type,
            **activation_params
        )
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "view"):
            s = x.size()
            def conv_view(x: torch.Tensor, f:int) -> torch.Tensor:
                return x.view(-1, f)
            self.view = functools.partial(conv_view, f=self.in_features)\
                if len(s) > 2 else torch.nn.Identity()
        return self.activation(self.linear(self.view(x)))