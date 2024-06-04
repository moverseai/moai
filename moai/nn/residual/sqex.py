import toolz
import torch

import moai.nn.activation as mia
import moai.nn.convolution as mic
from moai.nn.sqex import SqueezeExcite

__all__ = [
    "SqExBottleneck",
    "SqExPreResBottleneck",
    "SqExPreActivBottleneck",
]

__CONV_TO_SQUEEZE__ = {
    "conv2d": "average2d",
}

"""
    SqEx Bottleneck versions with 3 convolutions (2 projections, 1 bottleneck)    
"""


class SqExBottleneck(torch.nn.Module):
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super().__init__()
        conv_params = toolz.dissoc(convolution_params, "sqex")
        self.W1 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=bottleneck_features,
            stride=2 if strided else 1,
            **conv_params,
        )
        self.A1 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params,
        )
        self.W2 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=bottleneck_features,
            **conv_params,
        )
        self.A2 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params,
        )
        self.W3 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=out_features,
            **conv_params,
        )
        self.A3 = mia.make_activation(
            features=out_features, activation_type=activation_type, **activation_params
        )
        self.S = (
            torch.nn.Identity()
            if in_features == out_features
            else (
                mic.make_conv_1x1(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    **conv_params,
                    # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
                )
                if not strided
                else mic.make_conv_3x3(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    stride=2,
                    **conv_params,
                )
            )
        )
        self.SE = SqueezeExcite(
            features=out_features,
            squeeze={"type": __CONV_TO_SQUEEZE__[convolution_type]},
            excite={
                "type": "channel",
                "ratio": toolz.get_in(["sqex", "ratio"], activation_params, 0.2),
                "operation": {"type": "linear"},
                "activation": {"type": "relu"},
            },
            mode="mul",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W3(
            self.A2(self.W2(self.A1(self.W1(x))))
        )  # y = W3 * A2(W2 * A1(W1 * x))
        return self.A3(self.S(x) + self.SE(y))  # out = A3(S(x) + y)


class SqExPreResBottleneck(SqExBottleneck):
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super().__init__(
            convolution_type=convolution_type,
            activation_type=activation_type,
            in_features=in_features,
            out_features=out_features,
            bottleneck_features=bottleneck_features,
            convolution_params=convolution_params,
            activation_params=activation_params,
            strided=strided,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A3(
            self.W3(self.A2(self.W2(self.A1(self.W1(x)))))
        )  # y = A3(W3 * A2(W2 * A1(W1 * x)))
        return self.S(x) + self.SE(y)  # out = S(x) + y


class SqExPreActivBottleneck(torch.nn.Module):
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super().__init__()
        self.A1 = mia.make_activation(
            features=in_features, activation_type=activation_type, **activation_params
        )
        self.W1 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=bottleneck_features,
            stride=2 if strided else 1,
            **convolution_params,
        )
        self.A2 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params,
        )
        self.W2 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=bottleneck_features,
            **convolution_params,
        )
        self.A3 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params,
        )
        self.W3 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=out_features,
            **convolution_params,
        )
        self.S = (
            torch.nn.Identity()
            if in_features == out_features
            else (
                mic.make_conv_1x1(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    **convolution_params,
                    # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
                )
                if not strided
                else mic.make_conv_3x3(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    stride=2,
                    **convolution_params,
                )
            )
        )
        self.SE = SqueezeExcite(
            features=out_features,
            squeeze={"type": __CONV_TO_SQUEEZE__[convolution_type]},
            excite={
                "type": "channel",
                "ratio": toolz.get_in(["sqex", "ratio"], activation_params, 0.2),
                "operation": {"type": "linear"},
                "activation": {"type": "relu"},
            },
            mode="mul",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W3(
            self.A3(self.W2(self.A2(self.W1(self.A1(x)))))
        )  # y = W3 * A3(W2 * A2(W1 * A1(x)))
        return self.S(x) + self.SE(y)  # out = x + y
