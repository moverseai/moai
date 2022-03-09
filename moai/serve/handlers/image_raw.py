from collections.abc import Callable

import typing
import torch

__all__ = ['ImageRaw']

class ImageRaw(Callable):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> torch.Tensor:
        return {'color': torch.tensor(0.0).to(device)}