import functools
import typing
import torch
import logging

log = logging.getLogger(__name__)

__all__ = ["Mask"]

class Mask(torch.nn.Module):
    def __init__(self,
        value:      float=0.0,
        inverse:    bool=True,
    ):
        super(Mask, self).__init__()
        self.value = value
        self.inverse = inverse

    def forward(self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        m = ~mask if self.inverse else mask
        m_c = mask.shape[1]
        t_c = tensor.shape[1]
        if m_c > t_c:
            for i in range(m_c):
                tensor[m[:, i, ...]] = self.value
            tensor[~mask] = self.value
        elif t_c > m_c:
            for i in range(t_c):
                tensor[:, i, ...].unsqueeze(1)[m] = self.value
        else:
            tensor[m] = self.value
        return tensor

class Index(torch.nn.Module):
    def __init__(self,
        indices:    typing.Sequence[int],
        dim:        int=1,        
    ):
        super(Index, self).__init__()
        self.index = functools.partial(torch.index_select, dim=dim)
        self.register_buffer("indices", torch.tensor(list(indices), dtype=torch.long))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.index(input=tensor, index=self.indices)
