import torch

__all__ = [
    "Abs",
    "Scale",
    "Plus",
    "Minus",
    "Multiply",
    "PlusOne",
    "MinusOne",
    "MatrixTranspose",
    "Rad2Deg",
    "Deg2Rad",
    "Exponential",
]


class Abs(torch.nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class Scale(torch.nn.Module):
    def __init__(self, value: float):
        super(Scale, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.value


class Plus(torch.nn.Module):
    def __init__(self):
        super(Plus, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Minus(torch.nn.Module):
    def __init__(self):
        super(Minus, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y


class Multiply(torch.nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class PlusOne(torch.nn.Module):
    def __init__(self):
        super(PlusOne, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x + torch.ones_like(x)
        return x + 1.0


class MinusOne(torch.nn.Module):
    def __init__(self):
        super(MinusOne, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x - torch.ones_like(x)
        return x - 1.0


class MatrixTranspose(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.mT


class Deg2Rad(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, degrees: torch.Tensor) -> torch.Tensor:
        return torch.deg2rad(degrees)


class Rad2Deg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, radians: torch.Tensor) -> torch.Tensor:
        return torch.rad2deg(radians)


class Exponential(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(tensor)
