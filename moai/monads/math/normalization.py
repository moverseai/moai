import typing

import torch

from moai.monads.utils import expand_dims


class Znorm(torch.nn.Module):
    def __init__(self, dims: typing.Sequence[int]):
        super(Znorm, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std, mean = torch.std_mean(x, self.dims, keepdim=True)
        return (x - mean) / std


class MinMaxNorm(torch.nn.Module):
    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        super(MinMaxNorm, self).__init__()
        self.register_buffer("min", torch.scalar_tensor(min_value).float())
        self.register_buffer("max", torch.scalar_tensor(max_value).float())
        self.register_buffer("range", torch.scalar_tensor(self.max - self.min).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        mins, _ = torch.min(x.view(b, -1), dim=1)
        maxs, _ = torch.max(x.view(b, -1), dim=1)
        scale = self.range / (maxs - mins)
        # return x.sub(mins.view(b, 1, 1, 1))\
        #     .mul(scale.view(b, 1, 1, 1))\
        #     .add(self.min)
        return torch.addcmul(self.min, x - expand_dims(mins, x), expand_dims(scale, x))
