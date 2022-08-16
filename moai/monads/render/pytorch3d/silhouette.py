from moai.utils.arguments import assert_positive

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,    
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    TexturesVertex,
)

import torch
import logging
import numpy as np
import typing

__all__ = ['Silhouette']

log = logging.getLogger(__name__)

class Silhouette(torch.nn.Module):
    def __init__(self,        
        focal_length:       typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:    typing.Union[float, typing.Tuple[float, float]]=0.5,
        width:              int=None,
        height:             int=None,
        rotation:           torch.Tensor=None,
        translation:        torch.Tensor=None,
        z_reflection:       bool=True,
        faces_per_pixel:    int=1,
        blend_sigma:        float=1e-7,
        blend_gamma:        float=1e-7,
        raster_bin_size:    typing.Optional[int]=None,
        raster_blur_radius: float=1e-4,
        persistent:         bool=False,
        max_faces_per_bin:  typing.Optional[int]=None,
    ) -> None:
        super().__init__()
        self.focal_length = (focal_length, focal_length) if isinstance(focal_length, float) else focal_length
        self.principal_point = (principal_point, principal_point) if isinstance(principal_point, float) else principal_point
        self.resolution = (width, height)
        if principal_point is None:
            assert_positive(log, "width", width)
            assert_positive(log, "height", height)
            self.principal_point = (float(width // 2), float(height // 2))
            self.has_nominal_principal = True
        else:
            self.has_nominal_principal = False
        self.register_buffer('adjust_translation', 
            torch.Tensor([[1.0, 1.0, -1.0]]).float() if z_reflection else
            torch.ones(1, 3), persistent=persistent
        )
        if rotation is None:
             rotation = torch.eye(3)[np.newaxis, ...].float()
        rotation = torch.nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)
        if translation is None:
            translation = torch.zeros([1, 3]).float()
        translation = torch.nn.Parameter(translation, requires_grad=True)
        self.register_parameter('translation', translation)
        self.blend_params = BlendParams(sigma=blend_sigma, gamma=blend_gamma)
        self.faces_per_pixel = faces_per_pixel
        self.raster_blur_radius = raster_blur_radius
        self.raster_bin_size = raster_bin_size
        self.max_faces_per_bin = max_faces_per_bin

    def forward(self,
        vertices:           torch.Tensor, # [B, V, 3]
        faces:              torch.Tensor, # [B, F, 3]
        rotation:           torch.Tensor, # [B, 3, 3]
        translation:        torch.Tensor, # [B, 3]
        image:              torch.Tensor=None,
        nominal_image:      torch.Tensor=None,
    ):
        b = vertices.shape[0]
        T = translation if translation is not None else self.translation.expand(b, 3)
        T = T * self.adjust_translation        
        R = rotation if rotation is not None else self.rotation.expand(b, 3, 3)
        Rt = torch.zeros(b, 4, 4).to(T)
        Rt[:, :3, :3] = R
        Rt[:, 3, :3] = T
        Rt[:, 3, 3] = 1.0
        inv_Rt = torch.inverse(Rt)
        
        fx, fy = self.focal_length
        px, py = self.principal_point
        if image is not None:            
            w, h = image.shape[-1], image.shape[-2]
            px, py = w / 2.0, h / 2.0
        elif nominal_image is not None:
            w, h = self.resolution
            ow, oh = nominal_image.shape[-1], nominal_image.shape[-2]
            fx, fy = fx * (w / ow), fy * (h / oh)
            if not self.has_nominal_principal:
                px, py = (px / ow) * w, (py / oh) * h
            else:
                px, py = 0.5 * w, 0.5 * h            
        else:
            w, h = self.resolution
        cameras = PerspectiveCameras(
            [(fx, fy)], [(px, py)],  
            R=inv_Rt[:, :3, :3],  T=inv_Rt[:, 3, :3], 
            in_ndc=False, image_size=[(h, w)]
        ).to(vertices.device)
        raster_settings = RasterizationSettings(
            image_size=(h, w), 
            blur_radius=np.log(1. / self.raster_blur_radius - 1.) * self.blend_params.sigma, 
            faces_per_pixel=self.faces_per_pixel, bin_size=self.raster_bin_size,
            perspective_correct=True, max_faces_per_bin=self.max_faces_per_bin
        )

        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )        
        textures = TexturesVertex(verts_features=torch.ones_like(vertices)) # [B, V, 3]
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        color = silhouette_renderer(meshes_world=mesh)
        return color[:, np.newaxis, :, :, 3]