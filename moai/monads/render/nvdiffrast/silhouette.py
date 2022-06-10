from moai.monads.render.nvdiffrast import CONTEXT

import torch
import logging
import nvdiffrast.torch as dr

__all__ = ['Silhouette']

log = logging.getLogger(__name__)

class Silhouette(torch.nn.Module):    
    def __init__(self,
        width:                      int=512,
        height:                     int=256,
        position_gradient_scale:    float=1.0,
    ):
        super(Silhouette, self).__init__()
        self.resolution = [height, width]
        self.pos_grad_boost = position_gradient_scale

    def forward(self,
        ndc_vertices:       torch.Tensor,
        indices:            torch.Tensor,        
    ) -> torch.Tensor:
        f = indices[0] if len(indices.shape) > 2 else indices
        if f.dtype != torch.int32:
            f = f.int()
        rast_out, _ = dr.rasterize(CONTEXT, ndc_vertices, f, resolution=self.resolution)
        color   , _ = dr.interpolate(torch.ones_like(ndc_vertices[..., 0:1]), rast_out, f)
        color       = dr.antialias(color, rast_out, ndc_vertices, f)
        return color.permute(0, 3, 1, 2)#.contiguous()