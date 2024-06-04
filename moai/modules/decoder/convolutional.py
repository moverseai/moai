import typing

import omegaconf.omegaconf
import torch

import moai.nn.convolution as miconv
import moai.nn.sampling.spatial.upsample as miup

__all__ = ["Convolutional"]


class Convolutional(torch.nn.Module):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
        convolution: omegaconf.DictConfig,
        prediction: omegaconf.DictConfig,
        upscale: omegaconf.DictConfig,
    ):
        super(Convolutional, self).__init__()
        module_list = torch.nn.ModuleList()
        in_features, out_features = (
            configuration.bottleneck_features,
            configuration.out_features,
        )
        next_features = in_features // configuration.feature_block_divisor
        for block_size in configuration.blocks:
            module_list.append(
                miup.make_upsample(
                    upscale_type=upscale.type, features=in_features, **upscale.params
                )
            )
            for b in range(0, block_size):
                module_list.append(
                    miconv.make_conv_block(
                        block_type="conv2d",
                        convolution_type=convolution.type,
                        activation_type=convolution.activation.type,
                        in_features=in_features if b == 0 else next_features,
                        out_features=next_features,
                        convolution_params=convolution.params,
                        activation_params=convolution.activation.params,
                    )
                )
            in_features = next_features
            next_features //= configuration.feature_block_divisor
        module_list.append(
            miconv.make_conv_block(
                block_type="conv2d",  # TODO: check if possible to modify this
                convolution_type=prediction.convolution.type,  # TODO: check if this type can resolve sconv2d blocks
                activation_type=prediction.activation.type,
                in_features=in_features,
                out_features=out_features,
                convolution_params=prediction.convolution.params,
                activation_params=prediction.activation.params,
            )
        )
        self.sequential = torch.nn.Sequential(*module_list)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.sequential(features)
