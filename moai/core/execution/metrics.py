import inspect
import logging
import typing

import hydra.utils as hyu
import omegaconf.omegaconf
import toolz
import torch

import moai.core.execution.common as mic
from moai.core.execution.constants import Constants as C
from moai.validation.metric import MoaiMetric
from moai.validation.torchmetric import TorchMetric

log = logging.getLogger(__name__)

__all__ = ["Metrics"]


class Metrics(torch.nn.ModuleDict):

    def __init__(
        self,
        metrics: omegaconf.DictConfig = {},
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        errors = [k for k in kwargs if k not in metrics]
        if errors:
            log.error(
                f"The following metrics [{errors}] were not found in the configuration and will be ignored!"
            )
        self.reductions = []
        self.execs = []
        if not len(metrics):
            log.warning(
                "A collection of metrics is being used for validating the model, but no metrics have been assigned"
            )
        for i, key in enumerate(kwargs):
            if key not in metrics:
                log.warning(
                    f"Skipping metric `{key}` as it is not found in the configuration."
                )
                continue
            try:
                self.add_module(
                    key, hyu.instantiate(metrics[key])
                )  # TODO: stateless metrics can be re-used
            except Exception as e:
                log.error(f"Could not instantiate the metric '{key}': {e}")
                continue
            metric_kwargs = kwargs[key]
            metric = self[key]
            if isinstance(metric, MoaiMetric) or isinstance(metric, TorchMetric):
                sig = inspect.signature(metric.forward)
            else:
                log.warning(
                    f"Skipping metric `{key}` as it is neither a `MoaiMetric` or a `torchmetrics.Metric`."
                )
                continue
            sig_params = list(filter(lambda p: p in metric_kwargs, sig.parameters))
            extra_params = set(metric_kwargs.keys()) - set(sig_params) - set([C._OUT_])
            if extra_params:
                log.error(
                    f"The parameters [{extra_params}] are not part of the `{key}` metric signature ({list(sig.parameters.keys())})."
                )
            metric_kwargs = mic._dict_of_lists_to_list_of_dicts(metric_kwargs)
            for j, params in enumerate(metric_kwargs):
                self.execs.append((key, params[C._OUT_], toolz.dissoc(params, C._OUT_)))

    # NOTE: consider outputting per batch item metrics
    @torch.no_grad
    def forward(
        self, tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for key, out, kwargs in self.execs:
            metric = self[key](**toolz.valmap(lambda v: tensors[v], kwargs))
            tensors[f"{C._MOAI_METRICS_}.{out}"] = metric
