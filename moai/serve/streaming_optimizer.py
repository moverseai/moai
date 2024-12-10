import json
import logging
import os
import time
import typing
import zipfile
from collections import defaultdict

import benedict
import hydra.utils as hyu
import numpy as np
import toolz
import torch
import yaml
from hydra import compose, initialize
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.trainer import call
from toolz import merge_with
from ts.handler_utils.utils import send_intermediate_predict_response
from ts.metrics.metric_type_enum import MetricTypes

from moai.core.execution.constants import Constants as C
from moai.engine.callbacks.model import ModelCallbacks
from moai.utils.funcs import get_dict, get_list

_EXTRCT_FILES = False

if os.path.exists("conf.zip"):
    with zipfile.ZipFile("conf.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    with zipfile.ZipFile("src.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    _EXTRCT_FILES = True

# try:
#     from model import (
#         ModelServer,  # get model from local directory, otherwise configs could not be loaded
#     )
# except ImportError:
#     # needed for running archive correctly
#     from moai.serve.model import ModelServer

if _EXTRCT_FILES:
    from model import ModelServer
else:
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
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)

    def _get_overrides(self):
        overrides = []
        if os.path.exists("overrides.yaml"):
            with open("overrides.yaml") as f:
                overrides.extend(yaml.load(f, Loader=yaml.FullLoader))
                # return yaml.load(f, Loader=yaml.FullLoader)
        elif os.path.exists("streaming_handler_overrides.yaml"):
            with open("streaming_handler_overrides.yaml") as f:
                overrides.extend(yaml.load(f, Loader=yaml.FullLoader))
        return overrides
        # else:
        # return []

    def initialize(self, context):
        # call parent class initialize
        print(f"initializing streaming optimizer server with context: {context}")
        super().initialize(context, extract_files=~_EXTRCT_FILES)
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
                self.keys = toolz.get("keys", cfg.streaming_handlers, None)
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

        # NOTE:
        # remove run & rich callbacks as they are causing issues
        # self.trainer.callbacks.pop(0)
        # self.trainer.callbacks.pop(0)

    # @staticmethod
    # def stack_based_on_dim(arr):
    #     """
    #     Stacks using vstack for arrays with more than 1 dimension and hstack for 1-dimensional arrays.
    #     """
    #     if np.ndim(arr[0]) > 1:
    #         return np.vstack(arr)
    #     else:
    #         return np.hstack(arr)

    @staticmethod
    def stack_based_on_dim(arr):
        """
        Stacks using torch.stack for tensors.
        """
        if arr[0].dim() > 1:
            return torch.vstack(arr)
        else:
            return torch.hstack(arr)

    def handle(self, data: typing.Mapping[str, typing.Any], context: typing.Any):
        """
        Handle function responsible for returning an intermediate response.
        Used for streaming fit mode.
        """
        log.info("Streaming optimization handler called.")
        result = defaultdict(list)
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
        for batch_idx, batch in enumerate(torch.utils.data.DataLoader(dataloader)):
            self.model.optimization_step = 0
            # batch should be send to correct device
            batch = toolz.valmap(self.__to_device__, batch)
            batch = benedict.benedict(batch)
            # call on train batch start callbacks
            call._call_callback_hooks(
                self.trainer, "on_train_batch_start", batch, batch_idx
            )
            # call the training step
            self.model.training_step(batch, batch_idx)
            # send intermediate response
            key = 0  # DEBUG
            percent = float((batch_idx + 1) / len(dataloader) * 100)
            msg = json.dumps({"completed": percent}) + "\n"
            batch[key] = msg
            # batch[key] = (
            #     f"Running batch_idx {batch_idx} with completion percentage {float((batch_idx + 1)/len(dataloader) * 100):.2f}%."
            # )
            log.info(batch[key])
            send_intermediate_predict_response(
                batch,
                self.context.request_ids,
                "Intermediate response from the model.",
                200,
                self.context,
            )
            if self.keys:
                for k in self.keys:
                    result[k].append(batch[k])
            # calculate metrics on batch end
            # NOTE: we do not call default on_train_batch_end callback as it includes module logging operations which could cause errors
            if monitor := get_dict(self.model.monitor, f"{C._FIT_}.{C._BATCH_}"):
                for metric in get_list(monitor, C._METRICS_):
                    self.model.named_metrics[metric](batch)
                for tensor_monitor in get_list(monitor, C._MONITORS_):
                    extras = {
                        "stage": "fit",
                        "lightning_step": self.model.global_step,
                        "batch_idx": batch_idx,
                        "optimization_step": self.model.optimization_step,
                    }
                    self.model.named_monitors[tensor_monitor](batch, extras)
            # report metrics on batch end
            if C._MOAI_METRICS_ in batch:
                for key, val in batch[C._MOAI_METRICS_].items():
                    self.context.metrics.add_metric(
                        name=key,
                        value=float(val.detach().cpu().numpy()),
                        unit="value",
                        metric_type=MetricTypes.GAUGE,
                    )
            # report losses on batch end
            if C._MOAI_LOSSES_ in batch and "total" in batch[C._MOAI_LOSSES_]:
                for key, val in batch[C._MOAI_LOSSES_]["raw"].items():
                    self.context.metrics.add_metric(
                        name=key,
                        value=float(val.detach().cpu().numpy()),
                        unit="value",
                        metric_type=MetricTypes.GAUGE,
                    )
            # call on train batch end callbacks
            # call._call_callback_hooks(
            #     self.trainer,
            #     "on_train_batch_end",
            #     batch=batch,
            #     batch_idx=batch_idx,
            #     outputs=batch,
            # )
            call._call_callback_hooks(
                self.trainer,
                "on_train_batch_end",
                batch={
                    k: v
                    for k, v in batch.items()
                    if k not in {C._MOAI_LOSSES_, C._MOAI_METRICS_}
                },
                batch_idx=batch_idx,
                outputs={
                    k: v
                    for k, v in batch.items()
                    if k not in {C._MOAI_LOSSES_, C._MOAI_METRICS_}
                },
            )

        # call on epoch end callbacks
        call._call_callback_hooks(self.trainer, "on_train_epoch_end")
        # result = toolz.valmap(np.vstack, result)
        result = toolz.valmap(self.stack_based_on_dim, result)
        # result and original input data should be available in the post processing handler
        output = self.postprocess(toolz.merge(td, result))
        # TODO: add post processing handler
        stop_time = time.time()
        self.context.metrics.add_time(
            "FittingHandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )

        return output
