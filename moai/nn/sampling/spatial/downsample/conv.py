import moai.nn.convolution as mic

import torch

__all__ = [
    "StridedConv2d",
]

class StridedConv2d(torch.nn.Module):  #TODO: Add optional activation as well?
    def __init__(self,
        features: int,
        kernel_size: int=3,
        conv_type: str="conv2d",
        stride: int=2,
        padding: int=1
    ):
        super(StridedConv2d, self).__init__()
        self.conv = mic.make_conv_op(
            kernel_size=kernel_size,
            dilation=1,
            groups=1,
            conv_type=conv_type,
            in_channels=features,
            out_channels=features,
            bias=False,
            stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)