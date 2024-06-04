import torch

import moai.nn.activation as mia
import moai.nn.convolution as mic

__all__ = [
    "Standard",
    "PreResidual",
    "PreActivation",
]

"""
    Slightly adapted version of 
    Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    (adaptation on activation ordering as denoted in the factory below)
"""


class Standard(
    torch.nn.Module
):  # (b) in https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super(Standard, self).__init__()
        self.W1 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=out_features,
            stride=2 if strided else 1,
            **convolution_params
        )
        self.A1 = mia.make_activation(
            features=out_features, activation_type=activation_type, **activation_params
        )
        self.W2 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=out_features,
            out_channels=out_features,
            **convolution_params
        )
        self.A2 = mia.make_activation(
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
                    # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
                )
                if not strided
                else mic.make_conv_3x3(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    stride=2,
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W2(self.A1(self.W1(x)))  # y = W2 * A1(W1 * x)
        return self.A2(self.S(x) + y)  # out = A2(S(x) + y)


"""
    Slightly adapted version of
    Identity Mappings in Deep Residual Networks (https://arxiv.org/pdf/1603.05027.pdf)
    (adaptation on activation ordering as denoted in the factory below)
"""


class PreResidual(
    Standard
):  # (c) in https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super(PreResidual, self).__init__(
            convolution_type=convolution_type,
            activation_type=activation_type,
            in_features=in_features,
            out_features=out_features,
            convolution_params=convolution_params,
            activation_params=activation_params,
            strided=strided,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A2(self.W2(self.A1(self.W1(x))))  # y = A2(W2 * A1(W1 * x))
        return self.S(x) + y  # out = x + y


"""
    Slightly adapted version of
    Identity Mappings in Deep Residual Networks (https://arxiv.org/pdf/1603.05027.pdf)
    (adaptation on activation ordering as denoted in the factory below)
"""


class PreActivation(
    torch.nn.Module
):  # (e) in https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    def __init__(
        self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super(PreActivation, self).__init__()
        self.W1 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=out_features,
            stride=2 if strided else 1,
            **convolution_params
        )
        self.A1 = mia.make_activation(
            features=in_features, activation_type=activation_type, **activation_params
        )
        self.W2 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=out_features,
            out_channels=out_features,
            **convolution_params
        )
        self.A2 = mia.make_activation(
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
                    # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
                )
                if not strided
                else mic.make_conv_3x3(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    stride=2,
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W2(self.A2(self.W1(self.A1(x))))  # y = W2 * A2(W1 * A1(x))
        return self.S(x) + y  # out = x + y
