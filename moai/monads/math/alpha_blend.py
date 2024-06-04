import typing

import torch

__all__ = ["AlphaBlend"]


class AlphaBlend(torch.nn.Module):
    def __init__(self, blend: typing.Optional[float] = None) -> None:
        super().__init__()
        self.blend = blend

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        blend: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = blend if blend is not None else self.blend
        return left * b + right * (1.0 - b)
