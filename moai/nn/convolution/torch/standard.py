import torch

import moai.nn.activation as mia
import moai.nn.convolution as mic

__all__ = [
    "Conv2dBlock",
    "Conv1dBlock",
]


class Conv2dBlock(torch.nn.Module):
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        convolution_params: dict,
        activation_params: dict,
        **kwargs: dict
    ):
        super(Conv2dBlock, self).__init__()
        self.conv = mic.make_conv_op(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=out_features,
            **convolution_params  # TODO: either merge kwargs here or in the factory method
        )
        self.activation = mia.make_activation(
            features=out_features, activation_type=activation_type, **activation_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class Conv1dBlock(torch.nn.Module):
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        convolution_params: dict,
        activation_params: dict,
        **kwargs: dict
    ):
        super(Conv1dBlock, self).__init__()
        self.conv = mic.make_conv_op(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=out_features,
            **convolution_params  # TODO: either merge kwargs here or in the factory method
        )
        self.activation = mia.make_activation(
            features=out_features, activation_type=activation_type, **activation_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))
