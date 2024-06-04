import torch

import moai.nn.activation as mia
import moai.nn.deconvolution as mid

__all__ = [
    "Deconv2dBlock",
]


class Deconv2dBlock(torch.nn.Module):
    def __init__(
        self,
        deconvolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        deconvolution_params: dict,
        activation_params: dict,
        **kwargs: dict
    ):
        super(Deconv2dBlock, self).__init__()
        self.deconv = mid.make_deconv_op(
            deconvolution_type=deconvolution_type,
            in_channels=in_features,
            out_channels=out_features,
            **deconvolution_params
        )
        self.activation = mia.make_activation(
            features=out_features, activation_type=activation_type, **activation_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.deconv(x))
