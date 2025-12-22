import functools
import logging
import typing

import torch
from gsplat import rasterization, rasterization_2dgs

log = logging.getLogger(__name__)


class GaussianSplatRasterizer(torch.nn.Module):
    def __init__(
        self,
        model_type: typing.Literal["3dgs", "2dgs"] = "3dgs",
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        packed: bool = False,
        tile_size: int = 16,
        sh_degree: typing.Optional[int] = None,
        backgrounds: typing.Optional[torch.Tensor] = None,
        render_mode: typing.Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        sparse_grad: bool = False,
        absgrad: bool = False,
        distloss: bool = False,
        depth_mode: typing.Literal["expected", "median"] = "expected",
    ) -> None:
        super().__init__()
        model_type = model_type
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.radius_clip = radius_clip
        self.eps2d = eps2d
        self.packed = packed
        self.tile_size = tile_size
        self.sh_degree = sh_degree
        self.backgrounds = backgrounds
        self.render_mode = render_mode
        self.sparse_grad = sparse_grad
        self.absgrad = absgrad
        self.distloss = distloss
        self.depth_mode = depth_mode
        if model_type == "3dgs":
            self.rasteriser = functools.partial(
                rasterization,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                radius_clip=self.radius_clip,
                packed=self.packed,
                tile_size=self.tile_size,
                sh_degree=self.sh_degree,
                backgrounds=self.backgrounds,
                render_mode=self.render_mode,
                sparse_grad=self.sparse_grad,
                absgrad=self.absgrad,
                # distloss=self.distloss,
                # depth_mode=self.depth_mode,
            )
        elif model_type == "2dgs":
            self.rasteriser = functools.partial(
                rasterization_2dgs,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                radius_clip=self.radius_clip,
                eps2d=self.eps2d,
                sh_degree=self.sh_degree,
                packed=self.packed,
                tile_size=self.tile_size,
                backgrounds=self.backgrounds,
                render_mode=self.render_mode,
                sparse_grad=self.sparse_grad,
                absgrad=self.absgrad,
                distloss=self.distloss,
                depth_mode=self.depth_mode,
            )
        log.info(f"Initialized GaussianSplatRasterizer with model_type: {model_type}")

    def forward(
        self,
        means: torch.Tensor,  # [B, N, 3]
        quaternions: torch.Tensor,  # [B, N, 4]
        scales: torch.Tensor,  # [B, N, 3]
        opacities: torch.Tensor,  # [B, N]
        colors: torch.Tensor,  # [B, (C), N, D] or [B, (C,) N, K, 3],
        extrinsics: torch.Tensor,  # [B, C, 4, 4]
        intrinsics: torch.Tensor,  # [B, C, 3, 3]
        image: torch.Tensor,  # [B, (C), H, W] --> to get H, W dynamically
    ) -> typing.Dict[str, torch.Tensor]:
        # Placeholder implementation
        B = image.shape[0]
        H, W = image.shape[-2], image.shape[-1]
        if opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)

        rendered = self.rasteriser(
            means,
            quaternions,
            scales,
            opacities,
            colors,
            extrinsics,
            intrinsics,
            height=H,
            width=W,
        )
        # return rendered[0]  # return rendered image # B, C, W, H, 3
        return {
            "color": rendered[0],
            "alpha": rendered[1],
        }
