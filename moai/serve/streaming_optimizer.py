import logging
import os
import time
import typing
import zipfile

import hydra.utils as hyu
import numpy as np
import toolz
import torch
import tqdm
import yaml
from hydra.experimental import compose, initialize
from omegaconf.omegaconf import OmegaConf
from toolz import merge_with, valmap

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

    def _extract_files(self):
        with zipfile.ZipFile("conf.zip", "r") as conf_zip:
            conf_zip.extractall(".")
        with zipfile.ZipFile("src.zip", "r") as src_zip:
            src_zip.extractall(".")

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
        self.context = context
        start_time = time.time()

        self.trainer.strategy._lightning_module = (
            self.model
        )  # IMPORTANT: we need to set lightning module to the model
        self.model._trainer = self.trainer
        # self.model.configure_optimizers()
        self.trainer.strategy._lightning_optimizers = self.model.configure_optimizers()[
            0
        ]

        # TODO: add post processing handler
        stop_time = time.time()
        self.context.metrics.add_time(
            "FittingHandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
