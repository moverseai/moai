import typing

import torch

__all__ = ["Perspective"]


def _intrinsics_projection_matrix(
    fx: float,
    fy: float,
    px: float,
    py: float,
) -> torch.Tensor:
    return torch.Tensor(
        [
            [fx, 0.0, px],
            [0.0, fy, py],
            [0.0, 0.0, 1.0],
        ]
    ).float()


class Perspective(torch.nn.Module):  # NOTE: check if it is transposed
    def __init__(
        self,
        focal: typing.Optional[typing.Sequence[float]] = None,
        principal: typing.Optional[typing.Sequence[float]] = None,
        fov: float = 64.69,  # in degrees
        width: int = 320,
        height: int = 240,
        mode: str = "fov",  # one of 'intrinsics', 'fov'
    ):
        super(Perspective, self).__init__()
        if mode == "fov":
            tanfov = torch.tan((torch.deg2rad(torch.tensor(fov) / 2.0).float()))
            aspect_ratio = width / height
            fx, fy = -1.0 * width / 2.0 / tanfov, height / 2.0 / tanfov * aspect_ratio
            px, py = width / 2.0, height / 2.0
            intrinsics_matrix = _intrinsics_projection_matrix(fx, fy, px, py)
        elif mode == "intrinsics":
            fx, fy = focal
            px, py = principal if principal is not None else (width / 2.0, height / 2.0)
            intrinsics_matrix = _intrinsics_projection_matrix(fx, fy, px, py)
        self.register_buffer("intrinsics_matrix", intrinsics_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.intrinsics_matrix.expand(b, *self.intrinsics_matrix.shape[0:])
