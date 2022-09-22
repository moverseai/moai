import torch
import functools

__all__ = [
    'BN2d',
    'ReLu_BN',
    'BN2d_ReLu',
    'ReLu_BN2d',
    'LReLu_BN',
    'BN2d_LReLu',
    'LReLu_BN2d',
    'LReLu_Drop',
    'Normalize',
    'LReLu_BN_Drop',
    'LReLu_BN2d_Drop2d',
]

class BN2d(torch.nn.BatchNorm2d):
    def __init__(self,
        features: int,
        momentum: float=0.1,         
        epsilon: float=1e-5,
    ):
        super(BN2d, self).__init__(
            num_features=features, eps=epsilon, momentum=momentum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(BN2d, self).forward(x)

class ReLu_BN(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
    ):
        super(ReLu_BN, self).__init__()
        self.bn = torch.nn.BatchNorm1d(features, eps=epsilon)
        self.activation = torch.nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.activation(x))

class BN2d_ReLu(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
    ):
        super(BN2d_ReLu, self).__init__()
        self.bn = torch.nn.BatchNorm2d(features, eps=epsilon)
        self.activation = torch.nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(x))
        
class ReLu_BN2d(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
    ):
        super(ReLu_BN2d, self).__init__()
        self.bn = torch.nn.BatchNorm2d(features, eps=epsilon)
        self.activation = torch.nn.ReLU(inplace=inplace)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.activation(x))

class LReLu_BN(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
        negative_slope: float=0.01,
    ):
        super(LReLu_BN, self).__init__()
        self.bn = torch.nn.BatchNorm1d(features, eps=epsilon)
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.activation(x))

class LReLu_BN_Drop(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
        negative_slope:  float=0.01,
        p:  float=0.1,
    ):
        super(LReLu_BN_Drop, self).__init__()
        self.bn = torch.nn.BatchNorm1d(features, eps=epsilon)
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.bn(self.activation(x)))

class LReLu_Drop(torch.nn.Module):
    def __init__(self,
        inplace: bool=True,
        negative_slope:  float=0.01,
        p:  float=0.1,
    ):
        super(LReLu_BN_Drop, self).__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(x))

class BN2d_LReLu(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
        negative_slope: float=0.01,
    ):
        super(BN2d_LReLu, self).__init__()
        self.bn = torch.nn.BatchNorm2d(features, eps=epsilon)
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(x))

class LReLu_BN2d(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
        negative_slope: float=0.01,
    ):
        super(LReLu_BN2d, self).__init__()
        self.bn = torch.nn.BatchNorm2d(features, eps=epsilon)
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.activation(x))

class LReLu_BN2d_Drop2d(torch.nn.Module):
    def __init__(self,
        features: int,
        inplace: bool=True,
        epsilon: float=1e-5,
        negative_slope:  float=0.01,
        p:  float=0.1,
    ):
        super(LReLu_BN2d_Drop2d, self).__init__()
        self.bn = torch.nn.BatchNorm2d(features, eps=epsilon)
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        self.dropout = torch.nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.bn(self.activation(x)))

class Normalize(torch.nn.Module):
    def __init__(self,
        p:          int=2,
        dim:        int=1,
        epsilon:    float=1e-12,
    ):
        super(Normalize, self).__init__()
        self.func = functools.partial(torch.nn.functional.normalize,
            p=p, dim=dim, eps=epsilon,        
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)