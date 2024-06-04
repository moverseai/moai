import functools
import logging
import typing

import torch

import moai.nn.activation as mia
import moai.nn.utils as miu

log = logging.getLogger(__name__)

__all__ = [
    "make_deconv_op",
    "make_deconv_4x4",
    "make_deconv_block",
]

__DECONV_FACTORY__ = {
    "deconv2d": torch.nn.ConvTranspose2d,
}

# NOTE: https://stackoverflow.com/questions/50683039/conv2d-transpose-output-shape-using-formula


def _update_deconv_op(name: str, type: typing.Type):
    if name not in __DECONV_FACTORY__.keys():
        __DECONV_FACTORY__.update({name: type})
    else:
        log.error(
            f"Trying to add an already existing key {name} in the deconvolution operation factory."
        )


def make_deconv_op(
    deconvolution_type: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    **kwargs,
) -> torch.nn.Module:
    if deconvolution_type in __DECONV_FACTORY__.keys():
        return miu.instantiate(
            __DECONV_FACTORY__[deconvolution_type], **{**locals(), **kwargs}
        )
    else:
        log.error(f"Deconvolution type ({deconvolution_type}) not found.")


make_deconv_4x4 = functools.partial(
    make_deconv_op,  # NOTE: only pass deconv_type and inout features
    kernel_size=4,
    stride=2,
    dilation=1,
    padding=1,
    groups=1,
    bias=False,
)

__DECONV_BLOCK_FACTORY__ = {}


def _update_deconv_block(name: str, type: typing.Type):
    if name not in __DECONV_BLOCK_FACTORY__.keys():
        __DECONV_BLOCK_FACTORY__.update({name: type})
    else:
        log.error(
            f"Trying to add an already existing key {name} in the deconvolution block factory."
        )


def make_deconv_block(
    block_type: str,
    deconvolution_type: str,
    in_features: int,
    out_features: int,
    activation_type: str,
    deconvolution_params: dict = {"kernel_size": 4},
    activation_params: dict = {"inplace": True},
    **kwargs,
) -> torch.nn.Module:
    if block_type in __DECONV_BLOCK_FACTORY__.keys():
        return miu.instantiate(
            __DECONV_BLOCK_FACTORY__[block_type],
            **{
                **locals(),
                "deconvolution_params": deconvolution_params,
                "activation_params": activation_params,
                **kwargs,
            },
        )
    else:
        log.error(f"Deconvolutional block type ({block_type}) not found.")


import moai.nn.deconvolution.torch as mit

if "deconv2d" not in __DECONV_BLOCK_FACTORY__.keys():  # NOTE: DECONV OP
    _update_deconv_block("deconv2d", mit.Deconv2dBlock)

del mit

# NOTE: "coord_conv":
# NOTE: "partial_conv":

# NOTE: "coord_conv":
# NOTE: "partial_conv":
