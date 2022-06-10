import numpy as np
import torch
import typing

__all__ = ['Npy']

class Npy(torch.nn.Module):
    def __init__(self,
        **kwargs: typing.Mapping[str, str],        
    ) -> None:
        super().__init__()
        for k, f in kwargs.items():
            self.register_buffer(k,
                torch.from_numpy(np.load(f)).float()
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return { k:v for k, v in self.named_buffers() }