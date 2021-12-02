import torch
import typing

__all__ = [
    "Scalar",
    "Random",
    "Ones",
    "Zeros",
    "Clone",
]

class Scalar(torch.nn.Module):
    def __init__(self, 
        value: float
    ):
        super(Scalar, self).__init__()
        self.value = value
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.scalar_tensor(self.value,
            dtype=tensor.dtype, device=tensor.device
        )

class Random(torch.nn.Module):
    def __init__(self):
        super(Random, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # return torch.rand_like(tensor)
        return torch.rand(1, *tensor.shape[1:], 
            dtype=tensor.dtype, device=tensor.device).expand_as(tensor)

class Ones(torch.nn.Module):
    def __init__(self):
        super(Ones, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # return torch.ones_like(tensor)
        return torch.ones(1, *tensor.shape[1:], dtype=tensor.dtype,
                device=tensor.device).expand_as(tensor) if tensor.shape\
            else torch.scalar_tensor(1, dtype=tensor.dtype, device=tensor.device)

class Zeros(torch.nn.Module):
    def __init__(self):
        super(Zeros, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # return torch.zeros_like(tensor)
        return torch.zeros(1, *tensor.shape[1:], 
            dtype=tensor.dtype, device=tensor.device).expand_as(tensor)

class Clone(torch.nn.Module):
    def __init__(self):
        super(Clone, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

class Parameters(torch.nn.Module):
    def __init__(self,
        shape:          typing.Union[int, typing.Sequence[int]],
        init:           str='zeros', # one of [zeros, ones, rand, randn],
    ):
        super(Parameters, self).__init__()
        self.register_parameter('value', torch.nn.Parameter(
            getattr(torch, init)(tuple(shape))) #TODO: check omegaconf's convert type annotation
        )

    def forward(self, void: torch.Tensor) -> torch.nn.parameter.Parameter:
        return self.value
