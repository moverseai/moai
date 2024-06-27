import logging
import os
import time
import typing

import hydra.utils as hyu
import toolz
import torch
import yaml
from hydra.experimental import compose, initialize
from omegaconf.omegaconf import OmegaConf
from toolz import merge_with
from ts.metrics.metric_type_enum import MetricTypes

from moai.core.execution.constants import Constants as C
from moai.engine.callbacks.model import ModelCallbacks

try:
    from model import (
        ModelServer,  # get model from local directory, otherwise configs could not be loaded
    )
except ImportError:
    # needed for running archive correctly
    from moai.serve.model import ModelServer

log = logging.getLogger(__name__)

__all__ = ["StreamingOptimizerServer"]


def gather_tensors(dicts):
    """Merges dictionaries by gathering tensors into lists."""
    return merge_with(lambda x: x if isinstance(x, list) else [x], *dicts)


def concat_tensors(tensor_list):
    """Concatenates a list of tensors along dimension 0."""
    return torch.cat(tensor_list, dim=0)


def recursive_merge(dicts):
    """Recursively merges dictionaries."""

    def merge_fn(values):
        if all(isinstance(v, torch.Tensor) for v in values):
            return torch.cat(values, dim=0)
        elif all(isinstance(v, dict) for v in values):
            return recursive_merge(values)
        else:
            # Assumes all non-tensor, non-dict values are identical across dicts
            return values[0]

    return merge_with(merge_fn, *dicts)


class StreamingOptimizerServer(ModelServer):
    def __init__(self) -> None:
        super().__init__()

    def __to_device__(self, x):
        # merge values from dict using toolz
        if isinstance(x, dict):
            return toolz.valmap(self.__to_device__, x)
        elif isinstance(x, list):
            y = []
            for i in x:
                if isinstance(i, torch.Tensor):
                    y.append(i.to(self.dev))
                else:
                    pass
            return y
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)

    def _get_overrides(self):
        if os.path.exists("overrides.yaml"):
            with open("overrides.yaml") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        else:
            return []

    def initialize(self, context):
        # call parent class initialize
        super().initialize(context)
        main_conf = context.manifest["model"]["modelName"].replace("_", "/")
        # set model to training true before calling the training step
        self.model.train()
        try:
            with initialize(
                config_path="conf/" + "/".join(main_conf.split("/")[0:-1]),
                job_name=main_conf,
                version_base="1.3",
            ):
                cfg = compose(
                    config_name=main_conf.split("/")[-1],
                    overrides=self._get_overrides(),
                    return_hydra_config=True,
                )
                log.info("Loading trainer...")
                OmegaConf.set_struct(cfg, False)
                cfg.engine.runner.devices = [self.device.index]
                # TODO: check if device is set correctly
                OmegaConf.set_struct(cfg, True)
                self.trainer = hyu.instantiate(
                    cfg.engine.runner,
                    model_callbacks=ModelCallbacks(model=self.model),
                    _recursive_=False,
                )
        except Exception as e:
            log.error(f"An error has occured while loading the trainer:\n{e}")

    def handle(self, data: typing.Mapping[str, typing.Any], context: typing.Any):
        """
        Handle function responsible for returning an intermediate response.
        Used for streaming fit mode.
        """
        log.info("Streaming optimization handler called.")
        self.optimization_step = 0
        self.context = context
        start_time = time.time()

        self.trainer.strategy._lightning_module = (
            self.model
        )  # IMPORTANT: we need to set lightning module to the model
        self.model._trainer = self.trainer
        self.trainer.strategy._lightning_optimizers = self.model.configure_optimizers()[
            0
        ]
        # the preprocessing handlers
        # should return a dataset object to iterate over
        td = self.preprocess(data)
        # get the dataloader for returned dict
        dataloader = td["dataloader"]
        # iterate over the dataloader
        for batch, batch_idx in dataloader:
            # batch should be send to correct device
            batch = toolz.valmap(self.__to_device__, batch)
            # call the training step
            self.trainer.training_step(batch, batch_idx)
            self.optimization_step += 1
            # send intermediate response
            key = 0  # DEBUG
            batch[key] = (
                f"Running batch_idx {batch_idx} with completion percentage {batch_idx/len(dataloader)}."
            )
            # report metrics
            for key, val in batch[C._MOAI_METRICS_].items():
                self.context.metrics.add_metric(
                    name=key,
                    value=float(val.detach().cpu().numpy()),
                    unit="value",
                    metric_type=MetricTypes.GAUGE,
                )
            # report losses
            for key, val in batch[C._MOAI_LOSSES_]["raw"].items():
                self.context.metrics.add_metric(
                    name=key,
                    value=float(val.detach().cpu().numpy()),
                    unit="value",
                    metric_type=MetricTypes.GAUGE,
                )

        # TODO: add post processing handler
        stop_time = time.time()
        self.context.metrics.add_time(
            "FittingHandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
