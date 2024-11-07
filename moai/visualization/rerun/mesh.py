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
import trimesh

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
    log_seperate_frames: bool = False,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    num_frames, _, __ = vertices.shape
    if num_frames != faces.shape[0]:
        faces = np.repeat(np.expand_dims(faces, 0), num_frames, axis=0)
    for fr in range(num_frames):
        rr.set_time_sequence("frame", fr)
        o3d_mesh = trimesh.Trimesh(vertices=vertices[fr], faces=faces[fr])
        o3d_mesh.fix_normals()
        rr.log(
            path + f"/frame_{fr}" if log_seperate_frames else path,
            rr.Mesh3D(
                vertex_positions=vertices[fr],
                triangle_indices=faces[fr],
                vertex_colors=np.tile(
                    np.array(color.get_rgb() + (1,)), (vertices.shape[1], 1)
                ),
                vertex_normals=np.array(o3d_mesh.vertex_normals),
            ),
        )


def multiframe_points3d(
    points: np.ndarray,
    path: str,
    color: str,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
    log_seperate_frames: bool = False,
    radii: float = 0.02,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    num_frames, _, __ = points.shape
    for fr in range(num_frames):
        rr.set_time_sequence("frame", fr)
        rr.log(
            path + f"/frame_{fr}" if log_seperate_frames else path,
            rr.Points3D(
                positions=points[fr],
                colors=np.tile(
                    np.array(color.get_rgb() + (1,)), (points[fr].shape[0], 1)
                ),
                radii=radii,
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
    o3d_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces[0])
    o3d_mesh.fix_normals()
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=vertices[0],
            indices=faces[0],
            vertex_colors=np.tile(
                np.array(color.get_rgb() + (1,)), (vertices.shape[1], 1)
            ),  # TODO: memoize
            vertex_normals=np.array(o3d_mesh.vertex_normals),
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
