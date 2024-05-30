import logging
import typing
# import torch
import numpy as np
# import toolz
import colour

# import open3d as o3d

# from moai.utils.arguments import assert_numeric
# from moai.monads.execution.cascade import _create_accessor

log = logging.getLogger(__name__)

from collections.abc import Callable

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache
    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ['Mesh', 'mesh3d']

def mesh3d(
    vertices: np.ndarray, faces: np.ndarray, 
    path: str, color: str,
    optimization_step: typing.Optional[int]=None,
    lightning_step: typing.Optional[int]=None,
    iteration: typing.Optional[int]=None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    rr.log(path, rr.Mesh3D(
        vertex_positions=vertices, 
        indices=faces, 
        vertex_colors=np.tile(np.array(color.get_rgb() + (1,)), (vertices.shape[0], 1)) #TODO: memoize
    ))

class Mesh(Callable):
    def __init__(self, path: str, color: str):
        self.path = path
        self.color = colour.Color(color)

    def __call__(self, 
        vertices: np.ndarray, faces: np.ndarray, 
        optimization_step: typing.Optional[int]=None,
        lightning_step: typing.Optional[int]=None,
        iteration: typing.Optional[int]=None,
    ) -> None:
        if optimization_step is not None:
            rr.set_time_sequence("optimization_step", optimization_step)
        elif lightning_step is not None:
            rr.set_time_sequence("lightning_step", lightning_step)
        elif iteration is not None:
            rr.set_time_sequence("iteration", iteration)
        rr.log(self.path, rr.Mesh3D(
            vertex_positions=vertices, 
            indices=faces, 
            vertex_colors=np.tile(np.array(self.color.get_rgb() + (1,)), (vertices.shape[0], 1)) #TODO: memoize
        ))

# class _Mesh3d(Callable):
#     def __init__(
#         self,
#         name: str,  # Path to the points in the space hierarchy,
#         vertices: typing.Union[str, typing.Sequence[str]],
#         faces: typing.Union[str, typing.Sequence[str]],
#         color: typing.Union[str, typing.Sequence[str]],
#         path_to_hierarchy: typing.Sequence[str],
#         batch_percentage: float = 1.0,
#     ):
#         # rr.init(name)
#         # rr.spawn()
#         self.name = name
#         self.vertices = [vertices] if isinstance(vertices, str) else list(vertices)
#         self.vertex_accessors = [_create_accessor(k) for k in self.vertices]
#         self.faces = [faces] if isinstance(faces, str) else list(faces)
#         self.face_accessors = [_create_accessor(k) for k in self.faces]
#         self.path_to_hierarchy = path_to_hierarchy
#         self.colors = (
#             [colour.Color(color)]
#             if isinstance(color, str)
#             else list(colour.Color(c) for c in color)
#         )
#         self.batch_percentage = batch_percentage
#         assert_numeric(log, "batch percentage", batch_percentage, 0.0, 1.0)
#         # self.images = [image] if isinstance(image, str) else list(image)
#         self.access = lambda td, k: toolz.get_in(k.split("."), td)
#         self.step = 0

#     @staticmethod  # TODO: refactor these into a common module
#     def _passthrough(
#         tensors: typing.Dict[str, torch.Tensor],
#         key: str,
#         count: int,
#     ) -> torch.Tensor:
#         return tensors[key][:count] if key in tensors else None

#     def __call__(
#         self, tensors: typing.Dict[str, torch.Tensor], step: typing.Optional[int] = None
#     ) -> None:
#         for n, v, f, c, path in zip(
#             self.vertices,
#             self.vertex_accessors,
#             self.face_accessors,
#             self.colors,
#             self.path_to_hierarchy,
#         ):
#             rr.set_time_sequence("frame_nr", self.step)
#             vertices = v(tensors).detach().cpu().numpy()
#             faces = f(tensors).detach().cpu().numpy()
#             # compute normals
#             mesh = o3d.geometry.TriangleMesh(
#                 o3d.utility.Vector3dVector(vertices[0]),
#                 o3d.utility.Vector3iVector(faces[0]),
#             )
#             mesh.compute_vertex_normals()
#             rr.log_mesh(
#                 path, vertices, indices=faces, normals = np.asarray(mesh.vertex_normals), albedo_factor  = c.get_rgb()
#             )
#             # rr.log_points(
#             #     path,  # where to put the points in the hierarchy
#             #     self.access(tensors, k).detach().cpu().numpy()[0],
#             #     colors=c.get_rgb(),
#             # )
#         self.step += 1
