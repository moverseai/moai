import sys
import typing

import torch


# NOTE: assumes ldr images normalized in a specific range: set 1.0 for values in [0,255]
class PSNR(torch.nn.Module):
    def __init__(self, range_scale_factor: float = 255.0):
        super(PSNR, self).__init__()
        self.dr = range_scale_factor
        self.eps = torch.scalar_tensor(
            sys.float_info.epsilon
        )  # https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        mse = torch.mean(torch.sub(self.dr * gt, self.dr * pred) ** 2)
        return 10 * torch.log10((self.dr * self.dr) / (mse + self.eps.to(mse)))
