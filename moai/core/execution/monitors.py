import inspect
import logging
import typing

import hydra.utils as hyu
import omegaconf.omegaconf
import toolz
import torch

import moai.core.execution.common as mic
from moai.core.execution.constants import Constants as C

__all__ = ["Monitors"]

log = logging.getLogger(__name__)


class Monitors:
    def __init__(
        self,
        tensors: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ) -> None:
        self.operations = []
        if (
            tensors is None
        ):  # NOTE: corner case, make sure container predicates below dont fail
            tensors = {}
        for k in kwargs or {}:
            params = kwargs[k]  # NOTE: list args for multi calling
            override_params = params.get(C._PARAMS_, None) or {}
            target = tensors[k]
            if k in tensors:
                target = tensors[k]
            else:
                log.warning(f"Monitor `{k}` not found, skipping ...")
                continue
            operation = hyu.instantiate(target, **override_params)
            signature = inspect.signature(operation)
            extras = mic.__PRESET_ARGS__.intersection(signature.parameters.keys())
            # NOTE: operate should change to partial func call
            if mic._is_first_arg_tensor_dict(
                next(iter(signature.parameters.values()))
            ):  # signature.parameters[next(iter(args))]):
                args = (
                    set(toolz.drop(1, signature.parameters.keys()))
                    - mic.__PRESET_ARGS__
                )
                op_args = toolz.keyfilter(
                    lambda k: k in args, params
                )  # NOTE: currently assumes that no arg is named `params`
                self.operations.append(
                    mic.TensorsDictOperation(operation, op_args, extras)
                )
            else:
                args = signature.parameters.keys() - mic.__PRESET_ARGS__
                op_args = toolz.keyfilter(
                    lambda k: k in args, params
                )  # NOTE: currently assumes that no arg is named `params`
                tensor_args = mic._get_tensor_args(signature.parameters)
                self.operations.append(
                    mic.TensorsArgsOperation(
                        operation,
                        toolz.dissoc(op_args, *(args - tensor_args.keys())),
                        toolz.dissoc(op_args, *tensor_args.keys()),
                        extras,
                    )
                )

    @torch.no_grad
    def __call__(
        self,
        tensors: typing.Mapping[str, torch.Tensor],
        extras: typing.Mapping[str, typing.Any],
    ) -> None:
        for operation in self.operations:
            operation(tensors, extras)
