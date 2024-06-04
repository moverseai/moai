from moai.core.execution.constants import Constants as C
from collections import OrderedDict

import moai.core.execution.common as mic
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import toolz
import logging

log = logging.getLogger(__name__)

__all__ = ["Weighted"]

__REDUCTIONS__ = {
    'sum': torch.sum,
    'mean': torch.mean,
    'iou': lambda t: torch.mean(1.0 - (t / t[0].numel())), #NOTE: this reduction is only correct for batch size = 1?
}

class Weighted(torch.nn.ModuleDict):
    def __init__(self,
        objectives: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any]
    ):
        super().__init__()
        self.execs, self.weights, self.reductions, self.keyz = [], OrderedDict(), [], []
        if not len(objectives):
            log.warning("A weighted combination of objectives is being used for supervising the model, but no losses have been assigned.")
        if errors := [k for k in kwargs if k not in objectives]:
            log.error(f"Some objectives [orange bold strike blink] \[{errors}\] [/] were not found in the configuration and will be ignored!")        
        for i, key in enumerate(kwargs):
            if key not in objectives:
                log.warning(f"Skipping objective [red strike bold]`{key}`[/] as it is not found in the configuration :exclamation:")
                continue
            try:
                self.add_module(key, hyu.instantiate(objectives[key])) #TODO: stateless objectives can be re-used
            except Exception as e:
                log.error(f":excalamation: Could not instantiate the objective '{key}': {e}")
                continue
            objective_kwargs = kwargs[key]
            objective = self[key]
            sig = inspect.signature(objective.forward)
            sig_params = list(filter(lambda p: p in objective_kwargs, sig.parameters))            
            if extra_params := set(objective_kwargs.keys()) - set(sig_params) - set([C._OUT_, C._WEIGHT_, C._REDUCTION_]):
                log.error(f":warning: The parameters [bold yellow strike] \[{extra_params}\] [/] are not part of the `{key}` objective signature.")
            objective_kwargs = mic._dict_of_lists_to_list_of_dicts(objective_kwargs)
            for j, params in enumerate(objective_kwargs):
                if not (weight := toolz.get(C._WEIGHT_, params, default=None)):
                    log.warning(f":warning: Objective [yellow bold]`{key}`[/] has no assigned weights, automatically reverting to a weight of one (1.0).")
                    weight = 1.0
                if not (reduction := toolz.get(C._REDUCTION_, params, default=None)):
                    log.warning(f":warning: Objective [yellow bold]`{key}`[/] has no assigned reduction, automatically reverting to mean reduction.")
                    reduction = 'mean'
                self.execs.append((
                    key, params[C._OUT_], weight, reduction, 
                    toolz.dissoc(params, C._OUT_, C._WEIGHT_, C._REDUCTION_)
                ))

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        for i, (key, out, weight, reduction, kwargs) in enumerate(self.execs):
            error = self[key](**toolz.valmap(lambda v: tensors[v] if v is not None else v, kwargs))
            error = __REDUCTIONS__[reduction](error)
            tensors[f"{C._MOAI_LOSSES_}.raw.{out}"] = error
            tensors[f"{C._MOAI_LOSSES_}.weighted.{out}"] = weight * error
            if i == 0:
                tensors[f"{C._MOAI_LOSSES_}.total"] = tensors[f"{C._MOAI_LOSSES_}.weighted.{out}"]
            else:
                tensors[f"{C._MOAI_LOSSES_}.total"] += tensors[f"{C._MOAI_LOSSES_}.weighted.{out}"]