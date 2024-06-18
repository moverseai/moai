import typing

import numpy as np
import omegaconf.omegaconf
import toolz
import torch

__all__ = [
    "Scalar",
    "Random",
    "Ones",
    "Zeros",
    "Clone",
    "Parameter",
    "Parameters",
    "ZerosLike",
    "RandomLike",
    "OnesLike",
    "TemporalParams",
]


class Scalar(torch.nn.Module):
    def __init__(self, value: float):
        super(Scalar, self).__init__()
        self.value = value
        # TODO: make it a scalar buffer

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.scalar_tensor(self.value, dtype=tensor.dtype, device=tensor.device)


class Vector(torch.nn.Module):
    def __init__(
        self,
        shape: typing.Tuple[int],
        values: typing.Sequence[float],
    ) -> None:
        super(Vector, self).__init__()
        self.register_buffer(
            "vector",
            torch.from_numpy(np.array(values, dtype=np.float32)).reshape(tuple(shape)),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        b = tensor.shape[0]
        return self.vector.expand(b, *self.vector.shape)


class Random(torch.nn.Module):
    __RANDOMS__ = {
        "unit": torch.rand,
        "normal": torch.randn,
    }

    def __init__(
        self,
        shape: typing.Union[int, typing.Sequence[int]],
        scale: float = 1.0,
        mode: str = "unit",  # one of [unit, normal]
        includes_batch: bool = False,  # whether shape includes the batch dim
    ):
        super(Random, self).__init__()
        self.generate = Random.__RANDOMS__[mode]
        self.shape = shape if isinstance(shape, typing.Sequence) else [shape]
        self.scale = scale
        self.includes_batch = includes_batch

    def forward(self, tensor: torch.Tensor = None) -> torch.Tensor:
        shape = self.shape if self.includes_batch else [tensor.shape[0], *self.shape]
        device = tensor.device if tensor is not None else torch.device("cpu")
        generated = self.generate(tuple(shape), device=device)
        return generated * self.scale if self.scale != 1.0 else generated


class Ones(torch.nn.Module):
    def __init__(
        self,
        shape: typing.Union[int, typing.Sequence[int]],
        includes_batch: bool = False,  # whether shape includes the batch dim
    ):
        super(Ones, self).__init__()
        self.shape = shape if isinstance(shape, typing.Sequence) else [shape]
        self.includes_batch = includes_batch

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = self.shape if self.includes_batch else [tensor.shape[0], *self.shape]
        device = tensor.device if tensor is not None else torch.device("cpu")
        return torch.ones(shape, dtype=tensor.dtype, device=device)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.eye(
            tensor.shape[-1], dtype=tensor.dtype, device=tensor.device
        ).expand_as(tensor)


class Zeros(torch.nn.Module):
    def __init__(
        self,
        shape: typing.Union[int, typing.Sequence[int]],
        includes_batch: bool = False,  # whether shape includes the batch dim
    ):
        super(Zeros, self).__init__()
        self.shape = shape if isinstance(shape, typing.Sequence) else [shape]
        self.includes_batch = includes_batch

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = self.shape if self.includes_batch else [tensor.shape[0], *self.shape]
        device = tensor.device if tensor is not None else torch.device("cpu")
        return torch.zeros(*shape, dtype=tensor.dtype, device=device)


class ZerosLike(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # return torch.zeros_like(tensor)
        return torch.zeros_like(tensor)


class OnesLike(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # return torch.zeros_like(tensor)
        return torch.ones_like(tensor)


class RandomLike(torch.nn.Module):
    __RANDOMS__ = {
        "unit": torch.rand_like,
        "normal": torch.randn_like,
    }

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "unit",  # one of [unit, normal]
    ):
        super().__init__()
        self.generate = RandomLike.__RANDOMS__[mode]
        self.scale = scale

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        generated = self.generate(tensor)
        return generated * self.scale if self.scale != 1.0 else generated


class Clone(torch.nn.Module):
    def __init__(self):
        super(Clone, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()


class Parameter(torch.nn.Module):
    def __init__(
        self,
        shape: typing.Union[int, typing.Sequence[int]],
        init: str = "zeros",  # one of [zeros, ones, rand, randn],
    ):
        super(Parameter, self).__init__()
        if init == "eye":
            self.register_parameter(
                "value", torch.nn.Parameter(torch.eye(3).repeat(*shape))
            )
        else:
            self.register_parameter(
                "value",
                torch.nn.Parameter(
                    getattr(torch, init)(tuple(shape))
                ),  # TODO: check omegaconf's convert type annotation
            )

    def forward(self, void: torch.Tensor) -> torch.nn.parameter.Parameter:
        return self.value


class Parameters(torch.nn.Module):
    def __init__(
        self,
        parameters: omegaconf.DictConfig,
    ):
        super(Parameters, self).__init__()
        for name, param in parameters.items():
            self.register_parameter(
                str(name),
                torch.nn.Parameter(
                    getattr(torch, param.init or "zeros")(tuple(param.shape))
                ),  # TODO: check omegaconf's convert type annotation
            )

    def forward(self, void: torch.Tensor) -> torch.nn.parameter.Parameter:
        return dict(self.named_parameters())


class TemporalParams(torch.nn.Module):
    r"""
    Base class for temporal parameter generation.

    Args:
        parameters (omegaconf.DictConfig): parameters for temporal params
        window_size (int): number of frames to generate params for
    """

    def __init__(
        self,
        parameters: omegaconf.DictConfig,
        window_size: int,
    ) -> None:
        super().__init__()
        window_size = int(window_size) if isinstance(window_size, str) else window_size
        for name, param in parameters.items():
            # each param should be generated for each frame in the window
            for i in range(window_size):
                self.register_parameter(
                    str(name) + str(i),
                    torch.nn.Parameter(
                        getattr(torch, toolz.get("init", param, "zeros"))(
                            tuple(param.shape)
                        )
                    ),  # TODO: check omegaconf's convert type annotation
                )

        self.window_size = window_size

    def forward(self, void: torch.Tensor) -> torch.nn.parameter.Parameter:
        return dict(self.named_parameters())
