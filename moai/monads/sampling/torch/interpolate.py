from moai.utils.arguments import assert_choices
from moai.monads.utils.spatial import spatial_dims
from collections import namedtuple

import torch
import functools
import logging

log = logging.getLogger(__name__)

__all__ = [
    "Interpolate",
    "BilinearDownsample_x2",
    "NearestDownsample_x2",
    "BilinearDownsample",
    "NearestDownsample",
]

#NOTE: align corners doc: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/12

class Interpolate(torch.nn.Module):
    __MODES__ = ['nearest', 'linear', 'bilinear', 'area', 'bicubic', 'trilinear']

    STATIC_PARAMS = namedtuple('params', ['mode', 'align_corners', 'recompute_scale_factor'])

    def __init__(self,
        scale:                  float=0.0,
        width:                  int=-1,
        height:                 int=-1,
        #TODO: support depth as well for volumetric interpolation
        mode:                   str='bilinear', # one of ['nearest', 'linear', 'bilinear', 'area', 'bicubic', 'trilinear']            
        align_corners:          bool=False,
        recompute_scale_factor: bool=False,
        preserve_aspect_ratio:  bool=False,
    ):
        super(Interpolate, self).__init__()
        assert_choices(log, "interpolation mode", mode, Interpolate.__MODES__)
        if not scale > 0.0 and not (width > 0 and height > 0):
            log.error(f"Either scale ({scale}) or dimensions ([{width} x {height}]) need to be set for Interpolate.")
        self.func = functools.partial(
            torch.nn.functional.interpolate,
            scale_factor=scale, mode=mode, 
            align_corners=align_corners if mode != 'nearest' and mode != 'area' else None,
            recompute_scale_factor=recompute_scale_factor,
        ) if scale > 0.0 and width < 0 and height < 0 else\
            functools.partial(
                torch.nn.functional.interpolate,
                size=(height, width), mode=mode, 
                align_corners=align_corners if mode != 'nearest' and mode != 'area' else None,
                recompute_scale_factor=recompute_scale_factor,
            )
        self.params = Interpolate.STATIC_PARAMS(mode, align_corners, recompute_scale_factor)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.resolution = [height, width]

    def _calc_size(self, input_shape, target_shape):
        if self.preserve_aspect_ratio:
            h, w = target_shape
            if h > w:
                return h, int(input_shape[1] * (h / input_shape[0]))
            else:
                return int(input_shape[0] * (w / input_shape[1])), w
        else:
            return target_shape

    def forward(self,
        image:  torch.Tensor,
        target: torch.Tensor=None,
    ) -> torch.Tensor:
        return self.func(image) if target is None and not self.preserve_aspect_ratio\
            else torch.nn.functional.interpolate(
                image, size=self._calc_size(
                    image.shape[2:],
                    target.shape[2:] if target is not None else self.resolution
                ), **self.params._asdict()
            )#TODO: create a common func for spatial dims as tuple/list

BilinearDownsample_x2 = functools.partial(Interpolate,
    mode='bilinear', scale=0.5,
)
NearestDownsample_x2 = functools.partial(Interpolate,
    mode='nearest', scale=0.5,
)
BilinearSampling = functools.partial(Interpolate,
    mode='bilinear',
)
NearestSampling = functools.partial(Interpolate,
    mode='nearest',
)
NearestUpsample_x2 = functools.partial(Interpolate, 
    scale=2.0, mode='nearest'
)
BilinearUpsample_x2 = functools.partial(Interpolate,
    scale=2.0, mode='bilinear'
)