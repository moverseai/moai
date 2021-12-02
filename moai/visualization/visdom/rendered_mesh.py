from moai.visualization.visdom.image2d import Image2d
from moai.monads.execution.cascade import _create_accessor

import torch
import pyrender
import typing
import logging
import numpy as np
import math
import trimesh
import itertools
from PIL import Image

log = logging.getLogger(__name__)

__all__ = ["RenderedMesh"]

class RenderedMesh(Image2d):
    def __init__(self,
        vertices:           typing.Union[str, typing.Sequence[str]],
        faces:              typing.Union[str, typing.Sequence[str]],
        image:              typing.Union[str, typing.Sequence[str]],
        colormap:           typing.Union[str, typing.Sequence[str]],
        transform:          typing.Union[str, typing.Sequence[str]],
        translation:        typing.Union[str, typing.Sequence[str]]=None,
        rotation:           typing.Union[str, typing.Sequence[str]]=None,
        focal_length:       typing.Union[float, typing.Tuple[float, float]]=5000.0,
        scale:              float=1.0,
        batch_percentage:   float=1.0,
        name:               str="default",
        ip:                 str="http://localhost",
        port:               int=8097,
    ):
        super(RenderedMesh, self).__init__(
            image=image, name=name, ip=ip, port=port,
            type=list(itertools.repeat('color', len([vertices] if isinstance(vertices, str) else vertices))),
            transform=transform, batch_percentage=batch_percentage, colormap=colormap,
        )
        self.focal_length = (float(focal_length), float(focal_length)) \
            if isinstance(focal_length, float) or isinstance(focal_length, int) else focal_length
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0)
        )
        self.vertices = [vertices] if isinstance(vertices, str) else list(vertices)
        self.vertices = [_create_accessor(k) for k in self.vertices]
        self.faces = [faces] if isinstance(faces, str) else list(faces)
        self.faces = [_create_accessor(k) for k in self.faces]
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        for light in self._create_raymond_lights():
            self.scene.add_node(light)
        self.translation = list(itertools.repeat('', len(self.keys)) if translation is None else\
            ([translation] if isinstance(translation, str) else list(translation)))
        self.rotation = list(itertools.repeat('', len(self.keys)) if rotation is None else\
            ([rotation] if isinstance(rotation, str) else list(rotation)))
        self.scale = scale
        self.renderer = None

    def _get_renderer(self, width: int, height: int) -> pyrender.OffscreenRenderer:
        if self.renderer is None or self.renderer.viewport_width != width\
            or self.renderer.viewport_height != height:
                self.renderer = pyrender.OffscreenRenderer(
                    viewport_width=width, viewport_height=height, point_size=1.0
                )                    
        return self.renderer

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for v, f, r, t, k, _, tf, c in zip(
            self.vertices, self.faces, self.rotation, self.translation,
            self.keys, self.types, self.transforms, self.colormaps
        ):
            take = int(math.ceil(self.batch_percentage * tensors[k].shape[0]))
            background = self.colorize_map[c](
                self.transform_map[tf](tensors, k, take)
            )
            b, c, h, w = background.shape
            renderer = self._get_renderer(width=w, height=h)
            results = []
            for i in range(b):
                rotation = tensors[r][i].detach().cpu().numpy().squeeze() if r else np.eye(3)
                translation = tensors[t][i].detach().cpu().numpy().squeeze() if t else np.zeros(3)

                tmesh = trimesh.Trimesh(
                    v(tensors).detach().cpu().numpy().squeeze(),
                    f(tensors).detach().cpu().numpy().squeeze(),
                    process=False
                )
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                tmesh.apply_transform(rot)
                mesh = pyrender.Mesh.from_trimesh(tmesh, material=self.material)
                node = self.scene.add(mesh, 'mesh')

                # Equivalent to 180 degrees around the y-axis. Transforms the fit to
                # OpenGL compatible coordinate system.
                translation[0] *= -1.0
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = rotation
                camera_pose[:3, 3] = translation

                camera = pyrender.camera.IntrinsicsCamera(
                    fx=self.focal_length[0], cx=w // 2,
                    fy=self.focal_length[1], cy=h // 2,
                )
                cam = self.scene.add(camera, pose=camera_pose)

                color, _ = renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0

                valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
                input_img = background.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
                output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
                if self.scale != 1.0:
                    output_img = np.array(
                        Image.fromarray(
                            (output_img * 255.0).astype(np.uint8)
                        ).resize(
                            (int(w * self.scale), int(h * self.scale)), Image.ANTIALIAS
                        )
                    )
                results.append(output_img)
                self.scene.remove_node(node)
                self.scene.remove_node(cam)
            self.viz_map['color'](
                np.stack(results).transpose(0, 3, 1, 2),
                f"{k}_overlay", f"{k}_overlay", self.name
            )

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))

        return nodes
