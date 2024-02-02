from ts.torch_handler.base_handler import BaseHandler
from hydra.experimental import (
    compose,
    initialize,
)
from omegaconf.omegaconf import OmegaConf

import hydra.utils as hyu
import toolz
import os
import torch
import zipfile
import yaml
import logging
import typing

import numpy as np

import tqdm

log = logging.getLogger(__name__)

__all__ = ["StreamingOptimizerServer"]


class StreamingOptimizerServer(BaseHandler):
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
        properties = context.system_properties
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and not "FORCE_CPU" in os.environ:
            gpu_id = properties.get("gpu_id")
            if gpu_id is not None:
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cuda")
        log.info(f"Model set to run on a [{self.device}] device")
        # NOTE: IMPORTANT!!!! DEBUG WHILE TRAINING ONLY !!!
        # log.warning(f"IMPORTANT: Model explicitly set to CPU mode for debugging purposes! (was {self.device}).")
        # self.device = torch.device('cpu')
        # NOTE: IMPORTANT!!!! DEBUG WHILE TRAINING ONLY !!!
        main_conf = context.manifest["model"]["modelName"].replace("_", "/")
        log.info(f"Loading the {main_conf} endpoint.")
        self._extract_files()
        try:
            with initialize(config_path="conf", job_name=main_conf):
                cfg = compose(config_name=main_conf, overrides=self._get_overrides())
                self.optimizer = hyu.instantiate(cfg.model)
                self.engine = hyu.instantiate(cfg.engine)
        except Exception as e:
            log.error(f"An error has occured while loading the model:\n{e}")
        self.optimizer = self.optimizer.to(self.device)
        self.optimizer.eval()
        self.initialized = True
        log.info(f"Model ({type(self.optimizer.model)}) loaded successfully.")
        self.gradient_tolerance = cfg.fitter.gradient_tolerance
        self.relative_tolerance = cfg.fitter.relative_tolerance
        try:  # TODO: extract pre/post overrides to each merge as it currently crashes when finding overrides for pre/post that do not exist in the post/pre merged/instantiated config.
            handler_overrides = None
            if os.path.exists("handler_overrides.yaml"):
                with open("handler_overrides.yaml") as f:
                    handler_overrides = yaml.load(f, Loader=yaml.FullLoader)
                    log.debug(f"Loaded handler overrides:\n{handler_overrides}")
            with initialize(
                config_path="conf", job_name=f"{main_conf}_preprocess_handlers"
            ):
                cfg = compose(config_name="../pre")
                if (
                    handler_overrides is not None
                    and "preprocess" in handler_overrides.get("handlers", {})
                ):
                    cfg = OmegaConf.merge(cfg, {'handlers': {'preprocess': handler_overrides['handlers']['preprocess']}})
                    log.debug(f"Merged handler overrides:\n{cfg}")
                self.preproc = {
                    k: hyu.instantiate(h)
                    for k, h in (
                        toolz.get_in(["handlers", "preprocess"], cfg) or {}
                    ).items()
                }
            with initialize(
                config_path="conf", job_name=f"{main_conf}_postprocess_handlers"
            ):
                cfg = compose(config_name="../post")
                if (
                    handler_overrides is not None
                    and "postprocess" in handler_overrides.get("handlers", {})
                ):
                    cfg = OmegaConf.merge(cfg, {'handlers': {'postprocess': handler_overrides['handlers']['postprocess']}})
                    log.debug(f"Merged handler overrides:\n{cfg}")
                self.postproc = {
                    k: hyu.instantiate(h)
                    for k, h in (
                        toolz.get_in(["handlers", "postprocess"], cfg) or {}
                    ).items()
                }
        except Exception as e:
            log.error(f"An error has occured while loading the handlers:\n{e}")

    def preprocess(
        self,
        data: typing.Mapping[str, typing.Any],
    ) -> typing.Dict[str, torch.Tensor]:
        log.debug(f"Preprocessing input:\n{data}")
        # tensors = {"__moai__": {"json": data}}
        # body = data[0].get("body") or data[0].get("raw")
        # for k, p in self.preproc.items():
        # tensors = toolz.merge(tensors, p(body, self.device))
        # log.debug(f"Tensors: {tensors.keys()}")
        # return tensors
        return data

    def __to_device__(self, x):
        # merge values from dict using toolz
        if isinstance(x, dict):
            return toolz.valmap(self.__to_device__, x)
        elif isinstance(x, list):
            y = []
            for i in x:
                if isinstance(i, torch.Tensor):
                    y.append(i.to(self.device))
                else:
                    pass
            return y
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device).float()

    def relative_check(self, prev: torch.Tensor, current: torch.Tensor) -> float:
        relative_change = (prev - current) / max([prev.abs(), current.abs(), 1.0])
        return relative_change <= self.relative_tolerance

    def gradient_check(
        self, param_groups: typing.Sequence[typing.Dict[str, torch.nn.Parameter]]
    ) -> bool:
        return all(
            p.grad.view(-1).max().abs().item() < self.gradient_tolerance
            for p in toolz.concat((g["params"] for g in param_groups))
            if p.grad is not None
        )

    def is_any_param_nan(self, optimizer: torch.optim.Optimizer) -> bool:
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                if not torch.all(torch.isfinite(p)):
                    return True
        return

    def inference(
        self,
        data: typing.Mapping[str, torch.Tensor],
    ):
        log.info(f"Data:\n{data}")
        data = OmegaConf.create(data)[0]['body']
        log.info(f"Data:\n{type(data)}")
        # instantiate data
        test_iterator = hyu.instantiate(data["test"]["iterator"])
        test_loader = hyu.instantiate(data["test"]["loader"], test_iterator)
        pytl_callbacks = []
        pytl_callbacks.extend(
            [hyu.instantiate(c) for c in data["callbacks"].values()]
            if data["callbacks"] is not None
            else []
        )
        processed_data = []
        # do we need to do the below for each batch?
        for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
            print(f"Batch {batch_idx}")
            data = toolz.valmap(self.__to_device__, batch)
            # NOTE: debug only
            # data["joints_3d_predicted_filtered"] = data["landmarks"]["joints"][
            #     "filtered"
            # ].unsqueeze(0)
            # data["markers_3d_predicted_filtered"] = data["landmarks"]["markers"][
            #     "filtered"
            # ].unsqueeze(0)
            # NOTE: debug only
            # call batch start callbacks
            if pytl_callbacks is not None:
                for c in pytl_callbacks:
                    try:
                        on_train_batch_start = getattr(c, "on_train_batch_start")
                        on_train_batch_start(
                            trainer=None,
                            pl_module=self.optimizer,
                            batch=data,
                            batch_idx=batch_idx,
                            dataloader_idx=0,
                        )
                    except AttributeError:
                        log.debug(f"Callback {c} has no on_train_batch_start method.")
                        continue
                    # add test epoch end callback
                    try:
                        on_test_epoch_end = getattr(c, "on_test_epoch_end")
                        log.info(f"Calling on_test_epoch_end for {c}")
                        on_test_epoch_end(
                            trainer=None,
                            pl_module=self.optimizer,
                            outputs=None,
                        )
                    except AttributeError:
                        log.info(f"Callback {c} has no on_test_epoch_end method.")
                        continue
            self.last_loss = None
            if batch_idx == 0:
                # init only once for the first batch
                self.optimizer.initialize_parameters()
            optimizers, schedulers = self.optimizer.configure_optimizers()
            iters = list(toolz.mapcat(lambda o: o.iterations, toolz.unique(optimizers)))
            stages = list(toolz.mapcat(lambda o: o.name, toolz.unique(optimizers)))
            if self.optimizer.mode == "inference":
                with torch.no_grad():
                    self.optimizer.preprocess(data)
                    self.optimizer(data)
                    if batch_idx == 0:
                        # init only once for the first batch
                        self.optimizer.initialize(data)
            for i, (optim, iters, stage, sched) in enumerate(
                zip(optimizers, iters, stages, schedulers)
            ):
                log.info(f"Optimizing stage: {stage} for {iters} iterations")
                for p in self.optimizer.parameters():
                    p.requires_grad_(False)
                for pg in optim.param_groups:
                    for p in pg["params"]:
                        p.requires_grad_(True)

                def closure():
                    self.optimizer.optimizer_zero_grad(
                        epoch=0, batch_idx=0, optimizer=optim, optimizer_idx=i
                    )
                    self.loss = self.optimizer.training_step(
                        batch=data, batch_idx=0, optimizer_idx=i
                    )["loss"]
                    self.loss.backward()
                    self.optimizer.optimization_step += 1
                    data["__moai__"][
                        "optimization_step"
                    ] = self.optimizer.optimization_step
                    return self.loss

                for j in range(iters):
                    optim.step(closure=closure)
                    current_loss = self.loss
                    if hasattr(optim, "assign"):
                        with torch.no_grad():
                            optim.assign(data)
                    if (
                        (
                            self.last_loss is not None
                            and self.relative_check(self.last_loss, current_loss)
                        )
                        or self.gradient_check(optim.param_groups)
                        or not torch.isfinite(current_loss)
                        or self.is_any_param_nan(optim)
                    ):
                        log.warning(
                            f"Optimization stage '{stage}' stopped at iteration {j}/{iters}."
                        )
                        break
                    self.last_loss = current_loss
                sched.step()
                self.last_loss = None
                self.optimizer.optimization_step = 0
            metrics = self.optimizer.validation(data)
            print(f"Metrics: {metrics}")
            data["__moai__"]["batch_index"] = batch_idx
            processed_data.append(data)
            # NOTE: for debugging remove metrics
            for (
                k,
                v,
            ) in metrics.items():  # self.context is set in base handler's handle method
                self.context.metrics.add_metric(
                    name=k, value=float(v.detach().cpu().numpy()), unit="value"
                )
        return processed_data

    def postprocess(
        self, data: typing.Mapping[str, torch.Tensor]
    ) -> typing.Sequence[typing.Any]:
        outs = []
        for d in data:
            log.debug(f"Postprocessing outputs:\n{d['__moai__']}")
            # outs = []  # TODO: corner case with no postproc crashes, fix it
            for k, p in self.postproc.items():
                # res = p(d, d["__moai__"]["json"])
                res = p(d, d["__moai__"])
                if len(outs) == 0:
                    outs = res
                else:
                    for o, r in zip(outs, res):
                        o = toolz.merge(o, r)
        return outs

    # def handle(self, data, context):
    #     return super(data, context)
