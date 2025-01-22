import logging
import typing

import nvdiffrast.torch as dr
import torch

from moai.monads.render.nvdiffrast import CONTEXT

__all__ = ["Rasterize", "AttributeInterpolation"]

log = logging.getLogger(__name__)


class Rasterize(torch.nn.Module):
    def __init__(
        self,
        width: int = 512,
        height: int = 256,
        decomposed: bool = False,
    ):
        super().__init__()
        self.resolution = [height, width]
        self.decomposed = decomposed

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        ndc_vertices: torch.Tensor,
        indices: torch.Tensor,
        resolution_image: torch.Tensor = None,
    ) -> typing.Dict[str, torch.Tensor]:
        resolution = (
            self.resolution if resolution_image is None else resolution_image.shape[2:]
        )
        rasterized, derivatives = dr.rasterize(
            CONTEXT, ndc_vertices, indices, resolution=resolution
        )
        out = {
            "image": rasterized,
            "derivatives": derivatives,
        }
        if self.decomposed:
            out["triangles"] = {
                "barycentric": rasterized[..., :2],
                "id": rasterized[..., 3:],
            }
            out["normalized_depth"] = rasterized[..., 2:3]
        return out


class RasterizedForeground(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        triangle_ids: torch.Tensor,
        barycentric: typing.Optional[torch.Tensor] = None,
        normalized_depth: typing.Optional[torch.Tensor] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        nz_indices = torch.nonzero(triangle_ids[..., -1], as_tuple=True)
        out = {
            "indices": dict((str(i), t) for i, t in enumerate(nz_indices)),
            "triangles": {"id": triangle_ids[nz_indices]},
        }
        if barycentric is not None:
            out["triangles"]["barycentric"] = barycentric[nz_indices]
        if normalized_depth is not None:
            out["normalized_depth"] = normalized_depth[nz_indices]
        return out


class ForegroundRemap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        raster: torch.Tensor,
        foreground: torch.Tensor,
        indices: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        t = tuple(indices.values())
        raster[t] = foreground
        return raster


class AttributeInterpolation(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        attributes: torch.Tensor,
        rasterized: torch.Tensor,
        indices: torch.Tensor,
        derivatives: typing.Optional[torch.Tensor] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        attributes, derivatives = dr.interpolate(
            attributes, rasterized, indices, derivatives, "all"
        )
        return {
            "image": attributes,
            "derivatives": derivatives,
        }


class Antialias(torch.nn.Module):
    def __init__(
        self,
        position_gradient_scale: float = 1.0,
    ):
        super().__init__()
        self.pos_grad_boost = position_gradient_scale

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        attributes: torch.Tensor,
        rasterized: torch.Tensor,
        ndc_vertices: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: hash topology?
        antialiased_attributes = dr.antialias(
            attributes,
            rasterized,
            ndc_vertices,
            indices,
            pos_gradient_boost=self.pos_grad_boost,
        )
        return antialiased_attributes.permute(0, 3, 1, 2)  # .contiguous()
