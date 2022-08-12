from moai.utils.arguments import assert_numeric
from moai.monads.execution.cascade import _create_accessor
from collections.abc import Callable

import scenepic
import httpwatcher
import typing
import torch
import logging
import colour
import toolz
import math
import os
import numpy as np

log = logging.getLogger(__name__)

__all__ = ['Mesh']

class Mesh(Callable):    
    def __init__(self,
        vertices:         typing.Union[str, typing.Sequence[str]],
        faces:            typing.Union[str, typing.Sequence[str]],
        canvas:           typing.Union[int, typing.Sequence[int]],
        layer:            typing.Union[int, typing.Sequence[int]],
        color:            typing.Union[str, typing.Sequence[str]],
        batch_percentage: float=1.0,
        width:            int=600,
        height:           int=400,
        point_size:       float=0.1,
        name:             str="default",        
    ):
        self.name, self.point_size = name, point_size
        self.vertices = [vertices] if isinstance(vertices, str) else list(vertices)
        self.faces = [faces] if isinstance(faces, str) else list(faces)
        self.vertex_accessors = [_create_accessor(k) for k in self.vertices]        
        self.face_accesors = [(_create_accessor(k) if k else lambda _: None) for k in self.faces]
        self.ids = [canvas] if isinstance(canvas, int) else list(canvas)
        self.layers = [layer] if isinstance(layer, int) else list(layer)
        self.colors = [colour.Color(color)] if isinstance(color, str) else list(colour.Color(c) for c in color)
        self.batch_percentage = batch_percentage
        assert_numeric(log, 'batch percentage', batch_percentage, 0.0, 1.0)        
        self.WH = (width, height)
        scenepic_folder = os.path.join(os.getcwd(), 'scenepic')
        os.makedirs(scenepic_folder, exist_ok=True)
        log.info(f"Scenepic visualization enabled @ {scenepic_folder}.")
        log.warning(f"[scenepic]: For automatic refreshing @ `http://localhost:5555' use `httpwatcher -r {scenepic_folder}` (`pip install httpwatcher` if not available)")

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        meshes = {}
        scene = scenepic.Scene()
        canvases = [
            scene.create_canvas_3d(width=self.WH[0], height=self.WH[1]) 
            for _ in toolz.unique(self.ids)
        ]
        
        for n, v, f, c, l, id in zip(self.vertices, self.vertex_accessors,
            self.face_accesors, self.colors, self.layers, self.ids
        ):
            vertices = v(tensors).detach().cpu().numpy()
            faces = f(tensors)
            if faces is not None:
                faces = faces.detach().cpu().numpy()
            b = math.ceil(self.batch_percentage * vertices.shape[0])
            if id not in meshes:
                meshes[id] = []
            for i in range(b):
                mesh = scene.create_mesh(mesh_id=f"{n}_{i}",
                    shared_color=scenepic.Color(*c.rgb), layer_id=f"{l}",
                )
                if faces is not None:
                    mesh.add_mesh_without_normals(vertices[i], faces[i])
                else:
                    mesh.add_sphere()
                    mesh.apply_transform(scenepic.Transforms.Scale(self.point_size)) 
                    mesh.enable_instancing(positions=vertices[i]) 
                meshes[id].append(mesh)
        for k, m in meshes.items():
            frame = canvases[k].create_frame()
            grouped = toolz.groupby(lambda x: x.layer_id, m)
            for l, m in grouped.items():
                for j, mesh in enumerate(m):
                    xform = scenepic.Transforms.Translate(j * np.array([1.0, 0.0, 0.0]))
                    frame.add_mesh(mesh, xform)
            frame.set_layer_settings(dict(
                (str(n), {'opacity': 0.5}) for n in toolz.unique(self.layers)
            ))
        scene.link_canvas_events(*canvases)
        scene.save_as_html(os.path.join('scenepic', "index.html"), title=f"{self.name}")