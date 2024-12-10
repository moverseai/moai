import logging
import typing

import colour
import numpy as np
import torch

from moai.core.execution.constants import Constants as C

log = logging.getLogger(__name__)

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = [
    "optimization_losses",
    "scalar",
]


def scalar(
    value: np.ndarray,
    path: str,
    color: str,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
):
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    rr.log(
        path,
        rr.Scalar(float(value)),
    )


def optimization_losses(
    tensors: typing.Dict[str, torch.Tensor],
    path: str,
    keys: typing.Union[str, typing.Sequence[str]],
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
) -> None:
    if lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    for i, key in enumerate(keys):
        if key == "total":
            value = tensors.get(C._MOAI_LOSSES_TOTAL_)
        else:
            value = tensors.get(f"{C._MOAI_LOSSES_RAW_}.{key}")
        if value is not None:
            keypath = f"{path}/{key}"
            # if optimization_step == 1:
            #     rr.log(keypath, rr.SeriesLine(color=__COLORS[i], name=key), timeless=True)
            rr.log(keypath, rr.Scalar(float(value)))
