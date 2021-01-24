import moai.nn.convolution.spherical as mis

import torch

__all__ = [
    "SphericalConv2d",
]

class SphericalConv2d(mis.SphericalPad2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        dilation: int=1,
        stride: int=1,
        padding: int=0,
        groups: int=1,
        bias: bool=True
    ):
        super(SphericalConv2d, self).__init__(padding=padding if kernel_size > 1 else 0)
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            groups=groups,
            padding=0,
            padding_mode='zeros'
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO: if kernel_size == 1 dont pad
        padded = super(SphericalConv2d, self).forward(x)
        return self.conv2d(padded)