import itertools
import typing

import omegaconf.omegaconf
import tinycudann as tcnn
import torch

__all__ = ["EncodingMLP", "EncodingFullyFusedMLP"]

# NOTE: https://github.com/NVlabs/tiny-cuda-nn/issues/51#issuecomment-1054565404 // float16 vs float32
# NOTE: https://github.com/NVlabs/tiny-cuda-nn/issues/365 // hashgrid input range


class EncodingMLP(tcnn.Encoding):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: typing.Sequence[int],
        output_dims: int,
        encoding_config: omegaconf.DictConfig,
        seed: int = 1337,
    ):
        super().__init__(input_dims, dict(encoding_config), seed, dtype=torch.float32)
        net = [
            torch.nn.Linear(self.n_output_dims, hidden_dims[0], bias=True),
            torch.nn.ReLU(),
        ]
        # for h_in, h_out in itertools.pairwise(hidden_dims[1:]):
        for h_in, h_out in itertools.pairwise(hidden_dims):
            net.append(torch.nn.Linear(h_in, h_out, bias=True))
            net.append(torch.nn.ReLU())
        net.append(torch.nn.Linear(hidden_dims[-1], output_dims, bias=True))
        self.net = torch.nn.Sequential(*net)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        enc = super().forward(tensor.flatten(0, -2))
        x = self.net(enc)
        return x.view(*tensor.shape[:-1], x.shape[-1])


class EncodingFullyFusedMLP(tcnn.NetworkWithInputEncoding):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        encoding_config: omegaconf.DictConfig,
        network_config: omegaconf.DictConfig,
        seed: int = 1337,
    ):
        super().__init__(
            input_dims,
            output_dims,
            dict(encoding_config),
            dict(network_config),
            seed,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = super().forward(tensor.flatten(0, -2))
        return x.view(*tensor.shape[:-1], x.shape[-1]).float()
