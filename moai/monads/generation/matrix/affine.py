import torch
import typing
import roma
import numpy as np

__all__ = ["Rotation3D"]


def _create_rotation_x(roll: float):
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=torch.float32,
    )


def _create_rotation_y(pitch: float):
    return torch.tensor(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=torch.float32,
    )


def _create_rotation_z(yaw: float):
    return torch.tensor(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


class Rotation3D(torch.nn.Module):

    __CONSTRUCTORS__ = {
        "x": _create_rotation_x,
        "y": _create_rotation_y,
        "z": _create_rotation_z,
        "X": _create_rotation_x,
        "Y": _create_rotation_y,
        "Z": _create_rotation_z,
        "roll": _create_rotation_x,
        "pitch": _create_rotation_y,
        "yaw": _create_rotation_z,
    }

    def __init__(
        self,
        rotations: typing.Union[str, typing.Sequence[str]],  # angle@axis format
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        matrices = []
        for rot in rotations:
            euler, axis = rot.split("@")
            matrices.append(Rotation3D.__CONSTRUCTORS__[axis](np.deg2rad(float(euler))))
        self.register_buffer(
            "rotation",
            roma.rotmat_composition(matrices, normalize=False)
            if len(matrices) > 1
            else torch.cat(matrices),
        )
        self.rotation[self.rotation.abs() < epsilon] = 0.0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.rotation.expand(tensor.shape[0], 3, 3)
