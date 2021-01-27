import torch

class Relative(torch.nn.Module):
    def __init__(self,
        
    ):
        super(Relative, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y.inverse() @ x # from x -> to y 


class Inverse(torch.nn.Module):
    def __init__(self,
        
    ):
        super(Inverse, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.inverse()

class Transpose(torch.nn.Module):
    def __init__(self,
        
    ):
        super(Transpose, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, -1, -2)