import logging
import typing

import torch

import moai.nn.utils as miu

log = logging.getLogger(__name__)

__all__ = ["make_downsample"]

__DOWNSAMPLE_FACTORY__ = {
    "none": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "in_block": torch.nn.Identity,  # NOTE: needed for resnet
    "maxpool2d": torch.nn.MaxPool2d,
    "avgpool2d": torch.nn.AvgPool2d,
}


def _update_downsample_op(name: str, type: typing.Type):
    if name not in __DOWNSAMPLE_FACTORY__.keys():
        __DOWNSAMPLE_FACTORY__.update({name: type})
    else:
        log.error(
            f"Trying to add an already existing key {name} in the convolution operation factory."
        )


def make_downsample(
    downscale_type: str, features: int, kernel_size: int = 3, stride: int = 2, **kwargs
) -> torch.nn.Module:
    if downscale_type in __DOWNSAMPLE_FACTORY__.keys():
        return miu.instantiate(
            __DOWNSAMPLE_FACTORY__[downscale_type], **{**locals(), **kwargs}
        )
    else:
        log.error(f"Downscale type ({downscale_type}) not found.")


import moai.nn.sampling.spatial.downsample.conv as midsc

if "conv2d" not in __DOWNSAMPLE_FACTORY__.keys():
    _update_downsample_op("conv2d", midsc.StridedConv2d)

del midsc

import moai.nn.sampling.spatial.downsample.interpolate as midsi

if "downsample2d" not in __DOWNSAMPLE_FACTORY__.keys():
    _update_downsample_op("downsample2d", midsi.Downsample2d)

del midsi

import moai.nn.sampling.spatial.downsample.antialiased as midsaa

if "maxpool2d_aa" not in __DOWNSAMPLE_FACTORY__.keys():
    _update_downsample_op("maxpool2d_aa", midsaa.AntialiasedMaxPool2d)
    _update_downsample_op("conv2d_aa", midsaa.AntialiasedStridedConv2d)

del midsaa

import moai.nn.sampling.spatial.downsample.adaptive as midsad

if "adaptive" not in __DOWNSAMPLE_FACTORY__.keys():
    _update_downsample_op("adaptive", midsad.Adaptive)
    _update_downsample_op("adaptive1d", midsad.Adaptive1d)
    _update_downsample_op("adaptive2d", midsad.Adaptive2d)
    _update_downsample_op("adaptive3d", midsad.Adaptive3d)
    _update_downsample_op("adaptive1d_max", midsad.AdaptiveMaxPool1d)
    _update_downsample_op("adaptive2d_max", midsad.AdaptiveMaxPool2d)
    _update_downsample_op("adaptive3d_max", midsad.AdaptiveMaxPool3d)
    _update_downsample_op("adaptive1d_avg", midsad.AdaptiveAvgPool1d)
    _update_downsample_op("adaptive2d_avg", midsad.AdaptiveAvgPool2d)
    _update_downsample_op("adaptive3d_avg", midsad.AdaptiveAvgPool3d)
    _update_downsample_op("global1d_avg", midsad.GlobalAvgPool1d)
    _update_downsample_op("global2d_avg", midsad.GlobalAvgPool2d)
    _update_downsample_op("global3d_avg", midsad.GlobalAvgPool3d)

del midsad
