import moai.nn.utils as miu

import torch
import functools
import logging
import typing

log = logging.getLogger(__name__)

__all__ = [
    "make_upsample",
]

__UPSAMPLE_FACTORY__ = {
    "none":                 torch.nn.Identity,
}

def _update_upsample_op(name: str, type: typing.Type):
    if name not in __UPSAMPLE_FACTORY__.keys():
        __UPSAMPLE_FACTORY__.update({name: type})
    else:
        log.error(f"Trying to add an already existing key {name} in the convolution operation factory.")
 
def make_upsample(
    upscale_type: str,
    features: int,
    kernel_size: int=4,
    stride: int=2,
    **kwargs
) -> torch.nn.Module:
    if upscale_type in __UPSAMPLE_FACTORY__.keys():
        return miu.instantiate(__UPSAMPLE_FACTORY__[upscale_type], 
        **{
            **locals(),
            **kwargs 
        })
    else:
        log.error(f"Upscale type ({upscale_type}) not found.")

import moai.nn.sampling.spatial.upsample.deconv as mids

if "deconv2d" not in __UPSAMPLE_FACTORY__.keys():
    _update_upsample_op("deconv2d", mids.StridedDeconv2d)

del mids

import moai.nn.sampling.spatial.upsample.interpolate as miup

if "upsample2d" not in __UPSAMPLE_FACTORY__.keys():
    _update_upsample_op("upsample2d", miup.Upsample2d)

del miup