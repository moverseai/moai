import torch

import moai.nn.deconvolution as mid

__all__ = [
    "StridedDeconv2d",
]


class StridedDeconv2d(torch.nn.Module):
    def __init__(
        self,
        features: int,
        kernel_size: int = 4,
        deconv_type: str = "deconv2d",
        stride: int = 2,
        padding: int = 1,
    ):
        super(StridedDeconv2d, self).__init__()
        self.deconv = mid.make_deconv_op(
            deconvolution_type=deconv_type,
            in_channels=features,
            out_channels=features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)
