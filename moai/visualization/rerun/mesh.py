from collections.abc import Callable

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

import logging
import typing

import colour
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Mesh", "mesh3d", "multiframe_mesh3d"]


def multiframe_mesh3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
    color: str,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    num_frames, _, __ = vertices.shape
    for fr in range(num_frames):
        rr.log(
            path + f"/frame_{fr}",
            rr.Mesh3D(
                vertex_positions=vertices[fr],
                triangle_indices=faces[fr],
                vertex_colors=np.tile(
                    np.array(color.get_rgb() + (1,)), (vertices.shape[1], 1)
                ),
            ),
        )


def mesh3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
    color: str,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=vertices[0],
            indices=faces[0],
            vertex_colors=np.tile(
                np.array(color.get_rgb() + (1,)), (vertices.shape[1], 1)
            ),  # TODO: memoize
        ),
    )


class Mesh(Callable):
    def __init__(self, path: str, color: str):
        self.path = path
        self.color = colour.Color(color)

    def __call__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        optimization_step: typing.Optional[int] = None,
        lightning_step: typing.Optional[int] = None,
        iteration: typing.Optional[int] = None,
    ) -> None:
        if optimization_step is not None:
            rr.set_time_sequence("optimization_step", optimization_step)
        elif lightning_step is not None:
            rr.set_time_sequence("lightning_step", lightning_step)
        elif iteration is not None:
            rr.set_time_sequence("iteration", iteration)
        rr.log(
            self.path,
            rr.Mesh3D(
                vertex_positions=vertices[0],
                indices=faces[0],
                vertex_colors=np.tile(
                    np.array(self.color.get_rgb() + (1,)), (vertices.shape[1], 1)
                ),  # TODO: memoize
            ),
        )
