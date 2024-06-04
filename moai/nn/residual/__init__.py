import logging

import torch

import moai.nn.utils as miu

log = logging.getLogger(__name__)

__all__ = ["make_residual_block"]

import moai.nn.residual.bottleneck as mibtl
import moai.nn.residual.sqex as misebtl
import moai.nn.residual.standard as mistd

__BLOCK_FACTORY__ = {  # (b), (c) and (e) from https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    "standard": mistd.Standard,  # (b): y    =   A2(x  +   W2     *   A1(W1   *   x))
    "preresidual": mistd.PreResidual,  # (c): y    =   x    +   A2(W2   *   A1(W1   *   w))
    "preactivation": mistd.PreActivation,  # (e): y    =   x    +   W2      *   A2(W1   *   A1(x))
    # bottleneck versions
    "bottleneck": mibtl.Bottleneck,  # (b): y    =   A3(x    +   W3  *   A2(W2   *   A1(W1   *   x)))
    "preres_bottleneck": mibtl.PreResBottleneck,  # (c): y    =   x       +  A3(W3    *   A2(W2   *   A1(W1   *   x)))
    "preactiv_bottleneck": mibtl.PreActivBottleneck,  # (e): y    =   x       +  W3   *   A3(W2   *   A2(W1   *   A1(x)))
    "lambda_bottleneck": mibtl.LambdaBottleneck,
    # squeeze-n-excite bottleneck versions
    "sqex_bottleneck": misebtl.SqExBottleneck,  # (b): y    =   A3(x    +   SE(W3  *   A2(W2   *   A1(W1   *   x))))
    "sqex_preres_bottleneck": misebtl.SqExPreResBottleneck,  # (c): y    =   x       +  SE(A3(W3    *   A2(W2   *   A1(W1   *   x))))
    "sqex_preactiv_bottleneck": misebtl.SqExPreActivBottleneck,  # (e): y    =   x       +  SE(W3   *   A3(W2   *   A2(W1   *   A1(x))))
}

del mistd
del mibtl


def make_residual_block(
    block_type: str,
    convolution_type: str,
    in_features: int,
    out_features: int,
    bottleneck_features: int,
    activation_type: str,
    strided: bool,
    convolution_params: dict = {},
    downscale_params: dict = {},
    activation_params: dict = {"inplace": True},
    **kwargs
) -> torch.nn.Module:
    if block_type in __BLOCK_FACTORY__.keys():
        return miu.instantiate(
            __BLOCK_FACTORY__[block_type],
            **{
                **locals(),
                "convolution_params": convolution_params,
                "downscale_params": downscale_params,
                "activation_params": activation_params,
                **kwargs,
            }
        )
    else:
        log.error("Residual block type (%s) not found." % block_type)
