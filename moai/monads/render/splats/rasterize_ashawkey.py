import math
import typing

import numpy as np
import torch
from colour import Color
from diff_gaussian_rasterization_ashawkey import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

__all__ = ["GaussianSplatRasterizer"]


# NOTE: different gaussian splatting versions produce different number and order of outputs
#       but all collide using the same package


class GaussianSplatRasterizer(torch.nn.Module):
    def __init__(
        self,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        background_color: str = "black",
        prefiltered: bool = False,
        debug: bool = False,
        scale_modifier: float = 1.0,
    ):
        super().__init__()
        color = Color(background_color).get_rgb()
        self.register_buffer("background_color", torch.tensor(color))
        self.width, self.height = width, height
        self.prefiltered, self.debug = prefiltered, debug
        self.scale_modifier = scale_modifier

    def forward(
        self,
        view_matrix: torch.Tensor,  # [B, 4, 4]
        view_projection_matrix: torch.Tensor,  # [B, 4, 4]
        camera_position: torch.Tensor,  # [B, 3]
        positions: torch.Tensor,  # [B, V, 3]
        sh_coeffs: torch.Tensor,  # [B, SH, 3]
        opacities: torch.Tensor,  # [B, V, 1]
        rotations: torch.Tensor,  # [B, V, 4]
        scales: torch.Tensor,  # [B, V, 3]
        intrinsics: torch.Tensor,  # [B, 3, 3]
        image: typing.Optional[torch.Tensor] = None,  # [B, C, H, W]
        background_color: typing.Optional[torch.Tensor] = None,  # [B, 3]
    ):
        assert len(positions.shape) == 3
        B = view_matrix.shape[0]
        if positions.shape[0] != B:  # either many-to-many or one-to-many
            positions = positions.expand(B, -1, -1)
            sh_coeffs = sh_coeffs.expand(B, -1, -1, -1)
            opacities = opacities.expand(B, -1, -1)
            rotations = rotations.expand(B, -1, -1)
            scales = scales.expand(B, -1, -1)
        bg = background_color if background_color is not None else self.background_color
        sh_degree = math.sqrt(sh_coeffs.shape[-2]) - 1
        if bg.shape[0] != B:
            bg = bg.expand(B, -1)
        colors, radiis, depths, alphas = [], [], [], []
        for i in range(B):
            W = image[i].shape[-1] if image is not None else self.width
            H = image[i].shape[-2] if image is not None else self.height
            settings = GaussianRasterizationSettings(
                image_height=H,
                image_width=W,
                # tanfovx=2.0 * np.arctan(W / (2.0 * intrinsics[i, 0, 0].cpu().float())),
                # tanfovy=2.0 * np.arctan(H / (2.0 * intrinsics[i, 1, 1].cpu().float())),
                tanfovx=W / (2.0 * intrinsics[i, 0, 0].cpu().float()),
                tanfovy=H / (2.0 * intrinsics[i, 1, 1].cpu().float()),
                bg=bg[i],
                scale_modifier=self.scale_modifier,
                viewmatrix=view_matrix[i],
                projmatrix=view_projection_matrix[i],
                sh_degree=int(sh_degree),
                campos=camera_position[i],
                prefiltered=self.prefiltered,
                debug=self.debug,
            )
            screenspace_points = torch.zeros_like(
                positions[i]
            )  # , requires_grad=True) + 0
            screenspace_points.requires_grad_(True)
            screenspace_points.retain_grad()
            rasterizer = GaussianRasterizer(settings)
            color, radii, depth, alpha = rasterizer(  # TODO inv_depth
                means3D=positions[i],
                means2D=screenspace_points,
                opacities=opacities[i],
                scales=scales[i],
                rotations=rotations[i],
                shs=sh_coeffs[i],
                colors_precomp=None,
                cov3D_precomp=None,
            )
            colors.append(color)
            radiis.append(radii)
            depths.append(depth)
            alphas.append(alpha)
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
        return {
            "color": torch.stack(colors).clamp(0, 1),
            "radii": torch.stack(radiis),
            "depth": torch.stack(depths),
            "alpha": torch.stack(alphas),
        }
