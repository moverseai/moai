from moai.utils.arguments import (
    ensure_numeric_list,
    ensure_string_list,
)
from collections import OrderedDict
from moai.core.execution.constants import Constants

import moai.core.execution.common as mic
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import itertools
import toolz
import logging

log = logging.getLogger(__name__)

__all__ = ["Weighted"]

# from moai.monads.execution.cascade import _create_accessor

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
        self.execs, self.weights, self.reductions = [], OrderedDict(), [] # [], []
        if not len(objectives):
            log.warning("A weighted combination of objectives is being used for supervising the model, but no losses have been assigned.")
        errors = [k for k in kwargs if k not in objectives]
        if errors:
            log.error("Some objectives were not found in the configuration and will be ignored!")
        
        # loop = ((key, params) for key, params in kwargs.items() if key in objectives)
        
        self.keyz = []
        # for k, p in loop:
        for i, key in enumerate(kwargs):
            # self.add_module(k, hyu.instantiate(getattr(objectives, k)))
            if key not in objectives:
                log.warning(f"Skipping objective `{key}` as it is not found in the configuration.")
                continue
            try:
                self.add_module(key, hyu.instantiate(objectives[key])) #TODO: stateless objectives can be re-used
            except Exception as e:
                log.error(f"Could not instantiate the objective '{key}': {e}")
                continue
            # last_module = toolz.last(self.modules()) # moduledict is ordered
            # last_module = self[k]
            objective_kwargs = kwargs[key]
            objective = self[key]
            sig = inspect.signature(objective.forward)
            # objective_kwargs = toolz.valmap(ensure_string_list, objective_kwargs)
            sig_params = list(filter(lambda p: p in objective_kwargs, sig.parameters))
            extra_params = set(objective_kwargs.keys()) - set(sig_params) - set(['out', 'weight', 'reduction'])
            if extra_params:
                log.error(f"The parameters [{extra_params}] are not part of the `{key}` objective signature.")
            objective_kwargs = mic._dict_of_lists_to_list_of_dicts(objective_kwargs)
            for j, params in enumerate(objective_kwargs):
                weight = toolz.get('weight', params, default=None)
                if weight is None:
                    log.warning(f"Objective `{key}` has no assigned weights, automatically reverting to a weight of one (1.0).")
                    weight = 1.0
                reduction = toolz.get('reduction', params, default=None)
                if reduction is None:
                    log.warning(f"Objective `{key}` has no assigned reduction, automatically reverting to mean reduction.")
                    reduction = 'mean'
                self.execs.append((key, params['out'], weight, reduction, toolz.dissoc(params, 'out', 'weight', 'reduction')))

            # if 'weight' in p:
            #     wgts = iter(ensure_numeric_list(p['weight']))
            # else:
            #     log.warning(f"{k} loss has no assigned weights, automatically reverting to a weight of one (1.0).")
            #     wgts = itertools.cycle([1.0 / len(p['out'])])
            # if 'reduction' in p:
            #     reduction = iter(p['reduction'])
            # else:
            #     log.warning(f"{k} loss has no assigned reduction, automatically reverting to mean reduction.")
            #     reduction = itertools.cycle(['mean'])                
            #TODO: there is a bug if you pass in keys that are not bracketed ([]), i.e. as a list, even for a single arg
            # for keys in zip(*list(p[prop] for prop in itertools.chain(sig.parameters, ['out']) if p.get(prop) is not None)):
            #     accessors = [mic._create_accessor(k if isinstance(k, str) else toolz.get(0, k, None)) for k in keys[:-1]]
            #     self.execs.append(lambda tensor_dict, 
            #         acc=accessors, k=keys, p=sig.parameters.keys(), f=last_module:
            #         tensor_dict['losses']['raw'].update({
            #             k[-1]: f(**dict(zip(p, 
            #                 # list(tensor_dict[i] for i in k[:-1])
            #                 # list(tensor_dict.get(i, None) 
            #                 list(
            #                     a(tensor_dict) for a, i in zip(acc, k[:-1]) 
            #                     if i is not None or None
            #                 )
            #             )))
            #         })
            #     )
            #     self.keyz.append(keys[-1])
            #     # self.weights.append(next(wgts)) #TODO: error if no weight has been set? or implicit 1.0 ?
            #     self.weights[keys[-1]] = next(wgts)
            #     self.reductions.append(next(reduction))
            
    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # device = next(toolz.take(1, 
        #     filter(lambda t: isinstance(t, torch.Tensor), tensors.values())
        # )).device        
        # error = torch.tensor(0.0, dtype=torch.float32, device=device)
        # per_error_map = { }
        # # tensors['losses'] = {'raw': {}}
        # # for exe, w, k, r in zip(self.execs, self.weights, self.keyz, self.reductions):
        # for exe, (_, w), k, r in zip(self.execs, self.weights.items(), self.keyz, self.reductions):
        #     exe(tensors)
        #     # e = w * (torch.sum(tensors[k]) if r == 'sum' else torch.mean(tensors[k]))
        #     e = w * __REDUCTIONS__[r](tensors['losses']['raw'][k])
        #     per_error_map[k] = e
        #     error += e
        # tensors['losses']['weighted'] = per_error_map
        # tensors['losses']['total'] = error
        # return error, per_error_map
        for i, (key, out, weight, reduction, kwargs) in enumerate(self.execs):
            error = self[key](**toolz.valmap(lambda v: tensors[v] if v is not None else v, kwargs))
            error = __REDUCTIONS__[reduction](error)
            tensors[f"{Constants._MOAI_LOSSES_}.raw.{out}"] = error
            tensors[f"{Constants._MOAI_LOSSES_}.weighted.{out}"] = weight * error
            if i == 0:
                tensors[f"{Constants._MOAI_LOSSES_}.total"] = tensors[f"{Constants._MOAI_LOSSES_}.weighted.{out}"]
            else:
                tensors[f"{Constants._MOAI_LOSSES_}.total"] += tensors[f"{Constants._MOAI_LOSSES_}.weighted.{out}"]