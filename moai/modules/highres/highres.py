import functools
import typing

import omegaconf.omegaconf
import toolz
import torch

import moai.nn.activation as mia
import moai.nn.convolution as mic
import moai.nn.residual as mires
import moai.nn.sampling.spatial.upsample as mius

# NOTE: implementation adapted from https://github.com/HRNet/HRNet-Facial-Landmark-Detection

__all__ = ["HighResolution"]


class HighResolution(torch.nn.Module):
    def __init__(
        self,
        branches: int,
        depth: int,
        start_features: int,
        fuse: omegaconf.DictConfig,
        residual: omegaconf.DictConfig,
    ):
        super(HighResolution, self).__init__()
        self.num_branches = branches
        self.depth = depth
        self.branches = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[
                        mires.make_residual_block(
                            block_type=residual.type,
                            convolution_type=residual.convolution.type,
                            in_features=start_features * (2**b),
                            out_features=start_features * (2**b),
                            bottleneck_features=residual.bottleneck_features,
                            activation_type=residual.activation.type,
                            strided=False,
                            activation_params=residual.activation.params
                            or {"inplace": True},
                            convolution_params=toolz.merge(
                                residual.convolution.params or {},
                                {
                                    "bias": False,
                                },
                            ),
                        )
                        for d in range(self.depth)
                    ]
                )
                for b in range(self.num_branches)
            ]
        )
        self.fuse_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        (
                            torch.nn.Sequential(
                                *[
                                    mic.make_conv_block(
                                        block_type="conv2d",
                                        convolution_type=fuse.convolution,
                                        in_features=start_features * (2**j),
                                        out_features=start_features * (2**i),
                                        activation_type="none",
                                        convolution_params={
                                            "kernel_size": 1,
                                            "stride": 1,
                                            "padding": 0,
                                            "bias": False,
                                        },
                                    ),
                                    mius.make_upsample(
                                        upscale_type=fuse.upscale.type,
                                        stride=2 ** (j - i),
                                        features=start_features * (2**i),
                                        scale=2 ** (j - i),
                                    ),
                                ][:: (1 if fuse.upscale.conv_up else -1)]
                            )
                            if j > i
                            else (  # j < i -> j is lower res, we need to downscale
                                torch.nn.Sequential(
                                    *[
                                        mic.make_conv_block(
                                            block_type="conv2d",
                                            convolution_type=fuse.convolution,
                                            in_features=start_features * (2**j),
                                            out_features=(
                                                start_features * (2**i)
                                                if d == (i - j - 1)
                                                else start_features * (2**j)
                                            ),
                                            activation_type=(
                                                fuse.activation.prefusion
                                                if d == (i - j - 1)
                                                else fuse.activation.intermediate
                                            ),
                                            convolution_params={
                                                "kernel_size": 3,
                                                "stride": 2,
                                                "padding": 1,
                                                "bias": False,
                                            },
                                        )
                                        for d in range(i - j)
                                    ]
                                )
                                if j < i
                                else torch.nn.Identity()
                            )
                        )
                        for j in range(
                            self.num_branches
                        )  # each sequential operates on j scale
                    ]
                )
                for i in range(
                    self.num_branches
                )  # each module has sequentials that result in i scale
            ]
        )
        self.fuse_activ = mia.make_activation(
            activation_type=fuse.activation.final,
            features=sum(2**i for i in range(self.num_branches)),
            inplace=False,
        )

    def forward(
        self,
        x: typing.Union[typing.List[torch.Tensor], typing.Tuple[torch.Tensor, ...]],
    ) -> typing.List[torch.Tensor]:
        for i, (b_i, x_i) in enumerate(zip(self.branches, x)):
            x[i] = b_i(x_i)
        for i, f in enumerate(self.fuse_layers):
            x[i] = self.fuse_activ(
                functools.reduce(
                    lambda s, y: s + y, (f_i(x_i) for f_i, x_i in zip(f, x))
                )
            )
        return x
