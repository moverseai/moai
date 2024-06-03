from moai.utils.arguments import (
    ensure_path,
    ensure_choices,
)
from moai.monads.execution.cascade import _create_accessor

import trimesh
import torch
import typing
import logging
import os

__all__ = [
    "Mesh",
    "PointCloud"
]

log = logging.getLogger(__name__)

class Mesh(typing.Callable[[typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]], None]):

    __MODES__ = ['all', 'overwrite']
    __FORMATS__ = ['ply', 'obj']

    def __init__(self,
        path:              str,
        vertices:          typing.Union[str, typing.Sequence[str]],
        faces:             typing.Union[str, typing.Sequence[str]],
        filetype:          typing.Union[str, typing.Sequence[str]],
        mode:              str="overwrite", # all"
        format:            str="05d",
    ):
        self.mode = ensure_choices(log, "saving mode", mode, Mesh.__MODES__)
        self.folder = ensure_path(log, "output folder", path)
        self.formats = [ensure_choices(log, "output format", ext, Mesh.__FORMATS__) for ext in filetype]
        self.names = [vertices] if isinstance(vertices, str) else list(vertices)
        self.vertices = [_create_accessor(k) for k in self.names]
        self.faces = [faces] if isinstance(faces, str) else list(faces)
        self.faces = [_create_accessor(k) for k in self.faces]
        self.format = format
        log.info(f"Exporting meshes locally @ {self.folder}")
        self.index = 0

    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        for n, v, f, ext in zip(
            self.names, self.vertices, self.faces, self.formats
        ):
            verts = v(tensors).detach().cpu().numpy()
            indices = f(tensors).detach().cpu().numpy()
            bs = verts.shape[0]
            for i in range(bs):
                trimesh.Trimesh(
                    verts[i], indices[i], process=False
                ).export(os.path.join(
                    self.folder, f"{n}_{(self.index + i):{self.format}}.{ext}")
                )
        self.index = 0 if self.mode == "overwrite" else self.index + bs



class PointCloud(typing.Callable[[typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]], None]):

    __MODES__ = ['all', 'overwrite']
    __FORMATS__ = ['ply', 'obj']

    def __init__(self,
        path:              str,
        vertices:          typing.Union[str, typing.Sequence[str]],
        filetype:          typing.Union[str, typing.Sequence[str]],
        mode:              str="overwrite", # all"
        format:            str="05d",
    ):
        self.mode = ensure_choices(log, "saving mode", mode, PointCloud.__MODES__)
        self.folder = ensure_path(log, "output folder", path)
        self.formats = [ensure_choices(log, "output format", ext, PointCloud.__FORMATS__) for ext in filetype]
        self.names = [vertices] if isinstance(vertices, str) else list(vertices)
        self.vertices = [_create_accessor(k) for k in self.names]
        self.format = format
        log.info(f"Exporting pointclouds locally @ {self.folder}")
        self.index = 0

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for n, v, ext in zip(
            self.names, self.vertices, self.formats
        ):
            if v(tensors).shape[-1] is not 3:
                b, c, w, h = v(tensors).shape
                verts = v(tensors).permute(0,2,3,1).view(-1, w*h, c).detach().cpu().numpy()
            else:
                verts = v(tensors).detach().cpu().numpy()
            bs = verts.shape[0]
            for i in range(bs):
                trimesh.points.PointCloud(
                    verts[i], colors=None, metadata=None
                ).export(os.path.join(
                    self.folder, f"{n}_{(self.index + i):{self.format}}.{ext}")
                )
        self.index = 0 if self.mode == "overwrite" else self.index + bs

