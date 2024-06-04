import omegaconf.omegaconf
import torch

import moai.nn.convolution as mic

# NOTE: from [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999.pdf)

__all__ = ["BottleneckGate"]


class BottleneckGate(torch.nn.Module):
    def __init__(
        self,
        features: int,
        out_features: int,
        convolution: omegaconf.DictConfig,
        activation: omegaconf.DictConfig,
        blocks: int = 1,
    ):
        super(BottleneckGate, self).__init__()
        self.convs = torch.nn.Sequential(
            *[
                mic.make_conv_block(
                    block_type="conv2d",
                    convolution_type=convolution.type,
                    in_features=features,
                    out_features=out_features,
                    activation_type=activation.type,
                    convolution_params=convolution.params,
                    activation_params=activation.params,
                )
                for _ in range(blocks)
            ]
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        return self.convs(bottleneck)
