import inspect
import logging
import typing

import lark
import toolz
import torch

import moai.core.execution.common as mic
import moai.core.execution.expression as mie
from moai.core.execution.constants import Constants as C

log = logging.getLogger(__name__)

__all__ = ["Models"]


class Models(torch.nn.ModuleDict):
    execs: typing.List[typing.Callable]

    def __init__(
        self,
        models: torch.nn.ModuleDict,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        errors = [k for k in kwargs if k not in models]
        if errors:
            log.error(
                f"The following models [{errors}] were not found in the configuration and will be ignored!"
            )
        self.execs = []
        for i, key in enumerate(kwargs):
            if key not in models:
                log.warning(
                    "Skipping model '{key}' as it is not found in the configuration. This may lead to downstream errors."
                )
                continue
            module = models[key]
            model_params = kwargs[key]
            sig = inspect.signature(module.forward)
            sig_params = list(filter(lambda p: p in model_params, sig.parameters))
            extra_params = set(model_params.keys()) - set(sig_params) - set([C._OUT_])
            if extra_params:
                log.error(
                    f"The parameters [{extra_params}] are not part of the `{key}` model signature ({list(sig.parameters.keys())})."
                )
            model_params = mic._dict_of_lists_to_list_of_dicts(model_params)
            for j, params in enumerate(model_params):
                for k, v in params.items():
                    if isinstance(v, lark.Tree):
                        tmp_key = f"{key}{i}/{k}{j}"
                        self.add_module(tmp_key, mie.TreeModule(tmp_key, v))
                        self.execs.append((self[tmp_key], tmp_key, None))
                        params[k] = tmp_key
                self.execs.append(
                    (module, params[C._OUT_], toolz.dissoc(params, C._OUT_))
                )

    def forward(
        self, tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for module, out, kwargs in self.execs:
            if kwargs:
                tensors[out] = module(
                    **toolz.valmap(
                        lambda v: tensors[v] if v is not None else None, kwargs
                    )
                )
            else:
                tensors[out] = module(tensors)
        return tensors
