import typing

import omegaconf.omegaconf
import torch

import moai.nn.activation as mia
import moai.nn.convolution as mic
from moai.monads.sampling import Interpolate

# NOTE: from [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999.pdf)

__all__ = ["Attention"]


class Attention(torch.nn.Module):
    def __init__(
        self,
        features: int,
        gate_features: int,
        intermediate_features: typing.Union[int, float],
        skip_connection: omegaconf.DictConfig,
        gating: omegaconf.DictConfig,
        upsample: omegaconf.DictConfig,
        attention: omegaconf.DictConfig,
        projection: omegaconf.DictConfig,
    ):
        super().__init__()
        inter_features = intermediate_features or features
        inter_features = (
            int(inter_features * features)
            if isinstance(inter_features, float)
            else inter_features
        )
        inter_features = inter_features or 1
        self.skip = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=skip_connection.convolution.type,
            in_features=features,
            out_features=inter_features,
            activation_type="none",
            convolution_params=skip_connection.convolution.params,
        )
        self.gate = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=gating.convolution.type,
            in_features=gate_features,
            out_features=inter_features,
            activation_type="none",
            convolution_params=gating.convolution.params,
        )
        self.activation = mia.make_activation(
            activation_type=gating.activation.type,
            features=inter_features,
            **gating.activation.params
        )
        self.attention = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=attention.convolution.type,
            in_features=inter_features,
            out_features=1,
            activation_type=attention.activation.type,
            convolution_params=attention.convolution.params,
            activation_params=attention.activation.params,
        )
        self.proj = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=projection.convolution.type,
            in_features=features,
            out_features=features,
            activation_type=projection.activation.type,
            convolution_params=projection.convolution.params,
            activation_params=projection.activation.params,
        )
        self.upsample = Interpolate(
            scale=2.0,
            mode=upsample.mode,
            align_corners=upsample.align_corners,
            recompute_scale_factor=upsample.recompute_scale_factor,
        )

    def forward(
        self,
        enc: torch.Tensor,
        dec: torch.Tensor,
        gate: torch.Tensor = None,
    ) -> torch.Tensor:
        skip = self.skip(enc)
        gated = self.gate(gate)
        gated = self.upsample(gated, skip)
        activ = self.activation(skip + gated)
        attn = self.attention(activ)
        attn = self.upsample(attn, enc)
        proj = self.proj(attn * enc)
        return torch.cat([dec, proj], dim=1)
