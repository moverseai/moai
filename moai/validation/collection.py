import moai.core.execution.common as mic
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import toolz
import logging
import numpy as np
import benedict

from moai.validation.torchmetric import TorchMetric
from moai.validation.metric import MoaiMetric
from moai.core.execution.constants import Constants

log = logging.getLogger(__name__)

__all__ = ["Metrics"]

class Metrics(torch.nn.ModuleDict):
    # execs: typing.List[typing.Callable] = []

    # __REDUCTIONS__ = {
    #     "rmse": torch.sqrt,
    #     "mean": passthrough,
    #     "rad2deg": lambda x: torch.rad2deg(x).mean(),
    #     "dot2deg": lambda x: (90 - torch.rad2deg(_acos_safe(x))).abs().mean(),
    #     # 90 --> 0 conversion is required to map it to a distance metric in degrees
    #     # zero dot-distance applies for orthogonal vectors https://www.alamo.edu/contentassets/35e1aad11a064ee2ae161ba2ae3b2559/additional/math2412-dot-product.pdf
    # }

    def __init__(self,
        metrics: omegaconf.DictConfig = {},
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        errors = [k for k in kwargs if k not in metrics]
        if errors:
            log.error(f"The following metrics [{errors}] were not found in the configuration and will be ignored!")
        self.reductions = []
        self.execs = []
        if not len(metrics):
            log.warning("A collection of metrics is being used for validating the model, but no metrics have been assigned")
        for i, key in enumerate(kwargs):
            if key not in metrics:
                log.warning(f"Skipping metric `{key}` as it is not found in the configuration.")
                continue
            try:
                self.add_module(key, hyu.instantiate(metrics[key])) #TODO: stateless metrics can be re-used
            except Exception as e:
                log.error(f"Could not instantiate the metric '{key}': {e}")
                continue
            metric_kwargs = kwargs[key]
            metric = self[key]
            if isinstance(metric, MoaiMetric) or isinstance(metric, TorchMetric):
                sig = inspect.signature(metric.forward)            
            else:
                log.warning(f"Skipping metric `{key}` as it is neither a `MoaiMetric` or a `torchmetrics.Metric`.")
                continue
            sig_params = list(filter(lambda p: p in metric_kwargs, sig.parameters))
            extra_params = set(metric_kwargs.keys()) - set(sig_params) - set(['out'])
            if extra_params:
                log.error(f"The parameters [{extra_params}] are not part of the `{key}` metric signature.")
            metric_kwargs = mic._dict_of_lists_to_list_of_dicts(metric_kwargs)
            for j, params in enumerate(metric_kwargs):                
                self.execs.append((key, params['out'], toolz.dissoc(params, 'out')))

        # loop = ((key, params) for key, params in kwargs.items() if key in metrics)
        # # TODO: add message in case key is not found
        # for k, p in loop:
        #     self.add_module(k, hyu.instantiate(getattr(metrics, k)))
        #     # last_module = toolz.last(self.modules()) # moduledict is ordered
        #     last_module = self[k]
        #     sig = (
        #         inspect.signature(last_module.forward)
        #         if not isinstance(last_module, Metric)
        #         else inspect.signature(last_module.update)
        #     )
        #     if "out" not in p:
        #         p["out"] = [k]
        #     if "reduction" in p:
        #         reduction = iter(p["reduction"])
        #     else:
        #         log.warning(
        #             f"{k} metric has no assigned reduction, automatically reverting to mean reduction."
        #         )
        #         reduction = itertools.cycle(["mean"])
        #     for keys in zip(
        #         *toolz.remove(
        #             lambda x: not x,
        #             list(
        #                 p[prop]
        #                 for prop in itertools.chain(sig.parameters, ["out"])
        #                 if p.get(prop) is not None
        #             ),
        #         )
        #     ):
        #         accessors = [
        #             mic._create_accessor(
        #                 k if isinstance(k, str) else toolz.get(0, k, None)
        #             )
        #             for k in keys[:-1]
        #         ]
        #         self.execs.append(
        #             lambda tensor_dict, metric_dict, acc=accessors, k=keys, p=sig.parameters.keys(), f=last_module: metric_dict.update(
        #                 {
        #                     k[-1]: f(
        #                         **dict(
        #                             zip(
        #                                 p,
        #                                 list(
        #                                     a(tensor_dict)
        #                                     for a, i in zip(acc, k[:-1])
        #                                     if i is not None or None
        #                                 ),
        #                             )
        #                         )
        #                     )
        #                 }
        #             )
        #         )
        #         self.reductions.append(next(reduction))

    # NOTE: consider outputting per batch item metrics
    def forward(
        self, tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        # metrics = {}
        for key, out, kwargs in self.execs:
            # metric = self[key].update(**toolz.valmap(lambda v: tensors[v], kwargs))
            metric = self[key](**toolz.valmap(lambda v: tensors[v], kwargs))
            tensors[f"{Constants._MOAI_METRICS_}.{out}"] = metric
            # if metric is not None:
            #     tensors[f"{Constants._MOAI_METRICS_}.{out}"] = metric
            # else:
            #     tensors[f"{Constants._MOAI_METRICS_}.{out}"] = torch.empty(0)
        # tensors['_moai_._metrics_'].update(metrics)

        # for exe in self.execs:
        #     exe(tensors, metrics)
        # returned = {}
        # for i, (k, m) in enumerate(metrics.items()):
        #     if torch.is_tensor(m):
        #         returned[f"{k}"] = m
        #     elif isinstance(m, typing.Mapping):
        #         returned = toolz.merge(
        #             returned,
        #             toolz.keymap(
        #                 lambda x: f"{k}_{x}",
        #                 toolz.valmap(
        #                     lambda t: Metrics.__REDUCTIONS__[self.reductions[i]](t), m
        #                 ),
        #             ),
        #         )
        #     else:
        #         log.warning(
        #             f"Metric [{k}] return type ({type(m)} is not supported and is being ignored."
        #         )        
        # if toolz.get_in(["metrics"], tensors):
        #     returned = toolz.merge(tensors["metrics"], returned)
        # tensors["metrics"] = returned
        # return returned
    
    def aggregate(self, ) -> typing.Mapping[str, np.ndarray]:
        metrics = benedict.benedict({"{Constants._MOAI_METRICS_}": {}})
        for key, out, _ in self.execs:
            metrics[f"{Constants._MOAI_METRICS_}.{out}"] = self[key].compute()

