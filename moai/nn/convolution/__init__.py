import moai.nn.utils as miu
import moai.nn.activation as mia

import torch
from torch.nn import Identity as ID

import typing
import functools
import logging

log = logging.getLogger(__name__)

__all__ = [
    "make_conv_op",
    "make_conv_1x1",
    "make_conv_3x3",
    "make_conv_block",
]

__CONV_FACTORY__ = {
    "conv2d":           torch.nn.Conv2d,
    "conv1d":           torch.nn.Conv1d,
}

def _update_conv_op(name: str, type: typing.Type):
    if name not in __CONV_FACTORY__.keys():
        __CONV_FACTORY__.update({name: type})
    else:
        log.error(f"Trying to add an already existing key {name} in the convolution operation factory.")

def make_conv_op(
        convolution_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: int=1,
        dilation: int=1,        
        groups: int=1, 
        bias: bool=True,
        **kwargs
    ) -> torch.nn.Module:
        if convolution_type in __CONV_FACTORY__.keys():
            return miu.instantiate(__CONV_FACTORY__[convolution_type], 
            **{
                **locals(),
                 **kwargs
            })
        else:
            log.error(f"Convolution type {convolution_type} not found.")
            return ID()

make_conv_1x1 = functools.partial(make_conv_op, #NOTE: only pass conv_type and inout features
    kernel_size=1,
    stride=1,
    dilation=1,
    padding=0,
    groups=1,
    bias=False
)

make_conv_3x3 = functools.partial(make_conv_op, #NOTE: only pass conv_type and inout features
    kernel_size=3,
    stride=1,
    dilation=1,
    padding=1,
    groups=1,
    bias=True
)

__CONV_BLOCK_FACTORY__ = {

}

def _update_conv_block(name: str, type: typing.Type):
    if name not in __CONV_BLOCK_FACTORY__.keys():
        __CONV_BLOCK_FACTORY__.update({name: type})
    else:
        log.error(f"Trying to add an already existing key {name} in the convolution block factory.")

def make_conv_block(
    block_type: str,
    convolution_type: str,
    in_features: int,
    out_features: int,
    activation_type: str,
    convolution_params: dict={"kernel_size": 3},
    activation_params: dict={"inplace": True},
    **kwargs
) -> torch.nn.Module:
    if block_type in __CONV_BLOCK_FACTORY__.keys():
        return miu.instantiate(__CONV_BLOCK_FACTORY__[block_type],
            **{ 
                **locals(),
                "convolution_params": convolution_params, #TODO: merge /w kwargs?
                "activation_params": activation_params,
                **kwargs #TODO: merge to conv_params if otherwise unused?
            })
    else:
        log.error(f"Convolutional block type {block_type} not found.")
        return ID()

import moai.nn.convolution.torch as mit

if "conv2d" not in __CONV_BLOCK_FACTORY__.keys(): # CONV OP
    _update_conv_block("conv2d", mit.Conv2dBlock)
if "conv1d" not in __CONV_BLOCK_FACTORY__.keys(): # CONV OP
    _update_conv_block("conv1d", mit.Conv1dBlock)

del mit

#TODO: "coord_conv":
#TODO: "partial_conv":

import moai.nn.convolution.spherical as mis

if "sconv2d" not in __CONV_FACTORY__.keys():
    _update_conv_op("sconv2d", mis.SphericalConv2d)

#TODO: "coord_conv":
#TODO: "partial_conv":

from moai.nn.convolution.skeleton import SkeletonConvolution

if "skeleton" not in __CONV_FACTORY__.keys():
    _update_conv_op("skeleton", SkeletonConvolution)
