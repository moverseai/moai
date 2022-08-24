import torch.nn as ptnn
import moai.nn.utils as miu

import functools
import typing
import logging

log = logging.getLogger(__name__)

__all__ = [
    "make_linear_op",
    "make_linear_block",
    "make_linear_relu_bn",
    "make_linear_lrelu_bn",
    "make_linear_elu",
    "make_linear_sigmoid",
    "make_linear_tanh",
    "make_linear_no_activation",
]

__LINEAR_FACTORY__ = {
    "linear":               ptnn.Linear,
    "bilinear":             ptnn.Bilinear,
}

def _update_linear_op(name: str, type: typing.Type):
    if name not in __LINEAR_FACTORY__.keys():
        __LINEAR_FACTORY__.update({name: type})
    else:
        log.error(f"Trying to add an already existing key {name} in the linear operation factory.")

def make_linear_op(
    linear_type: str,
    in_features: int,
    out_features: int,
    bias: bool=True,
    **kwargs
) -> ptnn.Module:
    if linear_type in __LINEAR_FACTORY__.keys():
        other_kwargs = {}
        if linear_type == "bilinear" and "in1_features" not in kwargs:
            other_kwargs.update({"in1_features": in_features})
        return miu.instantiate(__LINEAR_FACTORY__[linear_type], 
        **{
            **locals(),
            **kwargs,
            **other_kwargs,
        })
    else:
        log.error(f"Linear type ({linear_type}) not found.")

__LINEAR_BLOCK_FACTORY__ = {
    
}

def _update_linear_block(name: str, type: typing.Type):
    if name not in __LINEAR_BLOCK_FACTORY__.keys():
        __LINEAR_BLOCK_FACTORY__.update({name: type})
    else:
        log.error(f"Trying to add an already existing key {name} in the linear block factory.")

def make_linear_block(
        block_type: str,
        linear_type: str,#TODO: obsolete, to remove
        in_features: int,
        out_features: int,
        activation_type: str,
        linear_params: dict={},
        activation_params: dict={"inplace": True},
        **kwargs
    ) -> ptnn.Module:
        if block_type in __LINEAR_BLOCK_FACTORY__.keys():
            return miu.instantiate(__LINEAR_BLOCK_FACTORY__[block_type],
                **{ 
                    **locals(),
                    "linear_params": linear_params, #TODO: merge /w kwargs?
                    "activation_params": activation_params,
                    **kwargs #TODO: merge to linear_params if otherwise unused?                    
                })
        else:
            log.error(f"Linear block type ({block_type}) not found.")

make_linear_relu_bn = functools.partial(make_linear_block,
    block_type="linear",
    linear_type="linear",
    activation_type="relu_bn",
)

make_linear_lrelu_bn = functools.partial(make_linear_block,
    block_type="linear",
    linear_type="linear",
    activation_type="lrelu_bn",
)

make_linear_lrelu = functools.partial(make_linear_block,
    block_type="linear",
    linear_type="linear",
    activation_type="lrelu",
)

make_linear_elu = functools.partial(make_linear_block,
    block_type="linear",
    linear_type="linear",
    activation_type="elu",
)

make_linear_sigmoid = functools.partial(make_linear_block,
    block_type="linear",
    linear_type="linear",
    activation_type="sigmoid",
)

make_linear_tanh = functools.partial(make_linear_block,
    block_type="linear",
    linear_type="linear",
    activation_type="tanh",
)

import moai.nn.linear.torch as ptln

if "linear" not in __LINEAR_BLOCK_FACTORY__:
    _update_linear_block("linear", ptln.LinearBlock)    