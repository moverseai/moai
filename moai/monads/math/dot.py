import torch
import string

__all__ = ['Dot']

class Dot(torch.nn.Module):
    def __init__(self, 
        dim:        int=-1,
        normalize:  bool=False,
    ) -> None:
        super().__init__()
        self.einstr = string. ascii_lowercase
        self.dim = dim
        self.normalize = normalize

    def forward(self, 
        x:      torch.Tensor,
        y:      torch.Tensor,
    ):
        dims = len(x.shape)
        lhs = self.einstr[:dims]
        rhs = lhs.replace(lhs[self.dim], '')
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=self.dim) 
            y = torch.nn.functional.normalize(y, dim=self.dim)
        return torch.einsum(f"{lhs},{lhs}->{rhs}", x, y)