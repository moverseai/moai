from collections import namedtuple
import torch
import typing
import logging
import numpy as np

__all__ = ['Silhouette']

log = logging.getLogger(__name__)

try:
    import pyredner
except:
    log.error(f"The pyredner package (`pip install pyredner`) is required to use the corresponding rendering monads.")

RenderParams = namedtuple('RenderParams', [
    'samples',
    'sample_pixel_center',
    'focal_length',
    'near_clip',
    'resolution',
    'principal_point',
    'opengl',
])

class Silhouette(torch.nn.Module):    
    def __init__(self,
        resolution:             typing.Sequence[int]=(256, 512),
        near_clip:              float=0.001,        
        samples:                typing.Union[int, typing.Sequence[int]]=(16, 4),
        sample_pixel_center:    bool=False,
        focal_length:           typing.Union[typing.Sequence[float], float]=5000.0,
        principal_point:        typing.Union[typing.Sequence[float], float]=None,
        opengl:                 bool=True,
    ):
        super(Silhouette, self).__init__()
        self.params = RenderParams(
            samples=(samples, samples) if isinstance(samples, int) else samples,
            resolution=resolution, opengl=opengl,
            sample_pixel_center=sample_pixel_center, near_clip=near_clip,
            focal_length=(focal_length, focal_length) if isinstance(focal_length, float) else focal_length, 
            principal_point=(principal_point, principal_point) if isinstance(principal_point, float) else principal_point
        )

    def forward(self,
        vertices:           torch.Tensor,
        indices:            torch.Tensor,
        uvs:                torch.Tensor=None,
        camera_translation: torch.Tensor=None,
        camera_rotation:    torch.Tensor=None,
        image:              torch.Tensor=None,
        intrinsics:         torch.Tensor=None,
    ) -> typing.Dict[str, torch.Tensor]:
        b, n, _ = vertices.shape
        translation = camera_translation if camera_translation is not None\
            else torch.zeros(b, 3, device=vertices.device)
        rotation = camera_rotation if camera_rotation is not None\
            else torch.eye(3, device=vertices.device)[np.newaxis, ...].expand(b, 3, 3)
        resolution = self.params.resolution or image.shape[-2:]
        if intrinsics is None:
            with torch.no_grad():
                ndc_width = 2.0 * resolution[-1]
                width_scale = resolution[-1] / image.shape[-1]
                height_scale = resolution[-2] / image.shape[-2]
                intrinsics = torch.eye(3, device=vertices.device)[np.newaxis, ...].expand(b, 3, 3)
                intrinsics[:, 0, 0] = (self.params.focal_length[0] / ndc_width) * width_scale
                intrinsics[:, 1, 1] = (self.params.focal_length[1] / ndc_width) * height_scale
                intrinsics[:, 0, 0] *= 4.0
                intrinsics[:, 1, 1] *= 4.0
                intrinsics[:, 0, 2] = (
                    (self.params.principal_point[0] - resolution[-1] / 2)
                    /  resolution[-1] / 2
                ) if self.params.principal_point is not None else 0.0 # resolution[-1] / 2
                intrinsics[:, 1, 2] = (
                    (self.params.principal_point[1] - resolution[-2] / 2)
                    / resolution[-2] / 2
                ) if self.params.principal_point is not None else 0.0 # resolution[-2] / 2
        kwargs = { 
            'material': pyredner.Material(
                diffuse_reflectance=torch.ones_like(translation[0]),
                use_vertex_color=True,
            )
        }
        scenes = []
        for i in range(b):
            f = indices[i].int().contiguous()
            v = vertices[i].contiguous()
            v[..., 0] = -1.0 * v[..., 0] # v3
            v[..., 1] = -1.0 * v[..., 1] # v3
            v[..., 2] = -1.0 * v[..., 2] # v3
            if uvs is not None:
                kwargs['uvs'] = uvs[i]
            else:
                kwargs['colors'] = torch.ones_like(v)    
            obj = pyredner.Object(vertices=v, indices=f,**kwargs)
            pos = translation[i]
            # if self.params.opengl:
                # # pos = pos * torch.tensor([-1.0, -1.0, 1.0]).to(pos)
                # pos = pos * torch.tensor([-1.0, 1.0, 1.0]).to(pos) # v3
            # up = (-1.0 if self.params.opengl else 1.0) * rotation[i][1]
            up = rotation[i][1] # v3
            fwd = (-1.0 if self.params.opengl else 1.0) * rotation[i][2]
            cam = pyredner.Camera(
                position=pos, up=up, look_at=pos + fwd,
                clip_near=self.params.near_clip,
                resolution=resolution,
                intrinsic_mat=intrinsics[i], 
                camera_type=pyredner.camera_type.perspective
            )
            scene = pyredner.Scene(camera=cam, objects=[obj])
            scenes.append(scene)
        img = pyredner.render_g_buffer(scene=scenes, 
            channels=[pyredner.channels.vertex_color], seed=None,                           
            num_samples=(self.params.samples[0], self.params.samples[1]),            
            sample_pixel_center=self.params.sample_pixel_center,
            device=vertices.device
        )
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.mean(dim=1, keepdim=True)
        if self.params.opengl:
            img = img.flip(dims=[-1])
        return torch.clamp(img, min=0.0, max=1.0)