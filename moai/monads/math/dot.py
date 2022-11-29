import torch
import string

__all__ = ['Dot']

class Dot(torch.nn.Module):
    def __init__(self, 
        dim:        int=-1,
        normalize:  bool=False,
        keep_dim:   bool=False,
    ) -> None:
        super().__init__()
        self.einstr = string. ascii_lowercase
        self.dim = dim
        self.normalize = normalize
        self.keep_dim = keep_dim

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
        ret = torch.einsum(f"{lhs},{lhs}->{rhs}", x, y)
        return ret.unsqueeze(self.dim) if self.keep_dim else ret