import functools
import logging
import time
import typing

import benedict
import hydra.utils as hyu
import toolz
import torch
from hydra import compose, initialize
from omegaconf.omegaconf import OmegaConf
from ts.metrics.metric_type_enum import MetricTypes

try:
    from model import (
        ModelServer,  # get model from local directory, otherwise configs could not be loaded
    )
except ImportError:
    # needed for running archive correctly
    from moai.serve.model import ModelServer

from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
from pytorch_lightning.trainer import call
from ts.handler_utils.utils import send_intermediate_predict_response

from moai.core.execution.constants import Constants as C
from moai.core.model import _create_assigner
from moai.engine.callbacks.model import ModelCallbacks
from moai.utils.funcs import get_list

log = logging.getLogger(__name__)  # NOTE: check name when logging from serve

__all__ = ["OptimizerServer"]

# import debugpy

# debugpy.listen(("0.0.0.0", 6789))
# debugpy.wait_for_client()


class OptimizerServer(ModelServer):
    def __init__(self) -> None:
        super().__init__()

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
        Used for fit mode.
        """
        log.info("Fitting handler called.")
        self.optimization_step = 0
        start_time = time.time()
        output = None  # DEBUG

        self.context = context

        batch = self.preprocess(data)
        log.info("Preprocessing done resulting in batch: ", batch)
        batch_idx = batch.get("batch_idx", 0)
        self.trainer.strategy._lightning_module = (
            self.model
        )  # IMPORTANT: we need to set lightning module to the model
        self.model._trainer = self.trainer
        # self.model.configure_optimizers()
        self.trainer.strategy._lightning_optimizers = self.model.configure_optimizers()[
            0
        ]

        # reimplement training step to get access to intermediate outputs (e.g. from intermediate iterations; this should be reported to the response)
        def closure(tensors, index, steps, stage, optimizer, objective):
            for step in steps:
                tensors = self.model.named_flows[step](tensors)
            self.model.named_objectives[objective](tensors)
            loss = tensors[f"{C._MOAI_LOSSES_}.total"]
            is_first_batch_to_accumulate = (
                index % self.trainer.accumulate_grad_batches == 0
            )
            if (
                self.trainer.accumulate_grad_batches == 1
                or not is_first_batch_to_accumulate
            ):
                call._call_callback_hooks(
                    self.trainer, "on_before_zero_grad", optimizer
                )
                call._call_lightning_module_hook(
                    self.trainer, "on_before_zero_grad", optimizer
                )
                call._call_lightning_module_hook(
                    self.trainer,
                    "optimizer_zero_grad",
                    self.trainer.current_epoch,
                    index,
                    optimizer,
                )
            call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
            self.optimization_step += 1
            if monitor := toolz.get_in(
                [C._FIT_, C._OPTIMIZATION_STEP_, stage], self.model.monitor
            ):
                should_monitor = (
                    self.optimization_step % monitor.get("_frequency_", 1) == 0
                )
                if (
                    tensor_monitor_steps := get_list(monitor, C._MONITORS_)
                ) and should_monitor:
                    with torch.no_grad():
                        for step in toolz.get_in([C._FLOWS_], monitor) or []:
                            self.model.named_flows[step](tensors)
                        extras = {
                            "lightning_step": self.model.global_step,
                            "epoch": self.model.current_epoch,
                            "optimization_step": self.optimization_step,
                            "batch_idx": batch_idx,
                            "stage": stage,
                        }
                        for step in tensor_monitor_steps:
                            self.model.named_monitors[step](tensors, extras)
            return loss

        batch = benedict.benedict(batch, keyattr_enabled=False)
        batch[C._MOAI_METRICS_] = {}
        batch[C._MOAI_LOSSES_] = {
            "raw": {},
            "weighted": {},
        }
        for stage, proc in self.model.process[C._FIT_][C._STAGES_].items():
            flows = proc[C._FLOWS_]
            objective = proc.get(C._OBJECTIVE_, None)
            assign_params = proc.get(C._ASSIGN_, None)
            if optim := proc.get(C._OPTIMIZER_, None):
                optimizers = self.model.optimizers()
                if isinstance(optimizers, list):
                    optimizer = optimizers[
                        list(self.model.named_optimizers.keys()).index(optim)
                    ]
                else:
                    if list(self.model.named_optimizers.keys()).index(optim) == 0:
                        optimizer = optimizers
                    else:
                        log.warning(
                            f"Optimizer {optim} with index {list(self.model.named_optimizers.keys()).index(optim)} is not found!"
                        )
            else:
                optimizer = None
            current_closure = functools.partial(
                closure, batch, batch_idx, flows, stage, optimizer, objective
            )
            for iter in range(proc.get(C._ITERATIONS_, 1)):
                if (  # when the strategy handles accumulation, we want to always call the optimizer step
                    not self.trainer.strategy.handles_gradient_accumulation
                    and self.trainer.fit_loop._should_accumulate()
                ):  # For gradient accumulation calculate loss (train step + train step end)
                    with _block_parallel_sync_behavior(
                        self.trainer.strategy, block=True
                    ):
                        current_closure()  # automatic_optimization=True: perform ddp sync only when performing optimizer_step
                else:
                    if optimizer is not None:
                        self.model.optimizer_step(
                            self.trainer.current_epoch,
                            batch_idx,
                            optimizer,
                            current_closure,
                        )
                        # create intermediate response
                        # key should be parseed
                        # from self.context.request_ids.keys()
                        key = 0  # DEBUG
                        max_iter = proc.get(C._ITERATIONS_, 1) - 1
                        batch[key] = (
                            f"Running Stage {stage} and Iteration {iter} from the model, with completion percentage {iter/max_iter}."
                        )
                        print(batch[key])
                        log.info(batch[key])
                        log.info("request_ids: ", self.context.request_ids)
                        send_intermediate_predict_response(
                            batch,
                            self.context.request_ids,
                            "Intermediate response from the model.",
                            200,
                            self.context,
                        )
                    else:  # NOTE: w/o an optim, it is a tensor setup step (e.g. inference)
                        with torch.no_grad():
                            for flow in flows:
                                batch = self.model.named_flows[flow](batch)
                                # create intermediate response
                                # key should be parseed
                                # from self.context.request_ids.keys()
                                key = 0  # DEBUG
                                batch[key] = (
                                    f"Running Stage {stage} and Iteration {iter} from the model."
                                )
                                send_intermediate_predict_response(
                                    batch,
                                    self.context.request_ids,
                                    "Intermediate response from the model.",
                                    200,
                                    self.context,
                                )
                                # TODO: this should be returned as intermediate response in the HTTP response
                if iter_monitor_stage := toolz.get_in(
                    [C._FIT_, C._STAGES_, stage], self.model.monitor
                ):
                    frequency = toolz.get(C._FREQUENCY_, iter_monitor_stage, 1)
                    should_monitor = iter % frequency == 0
                    # calculate metrics seperately
                    if (
                        iter_tensor_metric := iter_monitor_stage.get(C._METRICS_, None)
                    ) and should_monitor:
                        for metric in (
                            toolz.get(C._METRICS_, iter_monitor_stage, None) or []
                        ):
                            self.model.named_metrics[metric](batch)
                    if (
                        iter_tensor_monitor := iter_monitor_stage.get(
                            C._MONITORS_, None
                        )
                    ) and should_monitor:
                        for step in (
                            toolz.get(C._FLOWS_, iter_monitor_stage, None) or []
                        ):
                            self.model.named_flows[step](batch)
                        extras = {  # TODO: step => 'lightning_step'
                            "lightning_step": self.model.global_step,
                            "epoch": self.model.current_epoch,
                            "batch_idx": batch_idx,
                            "stage": stage,
                            "iteration": iter,
                        }
                        for step in iter_tensor_monitor:
                            self.model.named_monitors[step](batch, extras)
                        should_stop = False
                        for criterion in get_list(iter_monitor_stage, C._TERMINATION_):
                            if self.model.named_criteria[criterion](batch, extras):
                                should_stop = True
                                break
                        if should_stop:
                            log.info(
                                f"Terminating {stage} @ {iter} with criterion [{criterion}] !"
                            )
                            break
            # call the copy params for initialization
            if assign_params is not None:
                frequency = assign_params.get(
                    C._FREQUENCY_, 1
                )  # default to each batch end
                assigners = []
                if batch_idx == 0:  # get initializers only in the first batch
                    for i, o in assign_params.items():
                        if (
                            i == C._FREQUENCY_
                        ):  # NOTE: refactor this, keys should be not coupled like this
                            continue
                        assigners.append((i, _create_assigner(o)))
                with torch.no_grad():  # use torch no grad as most params are leaf tensors and assign is an inplace operation
                    if frequency == 0:  # if frequency is 0 call only once
                        if batch_idx == 0:
                            self.model._assign_params(assigners, batch)
                    else:
                        if batch_idx % frequency == 0:
                            self.model._assign_params(assigners, batch)

        # TODO: add post processing handler
        stop_time = time.time()

        # TODO: add post processing handler
        stop_time = time.time()
        self.context.metrics.add_time(
            "FittingHandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        # NOTE: Do we need to call on batch end to calculate metrics?
        # self.model.on_train_batch_end(outputs, batch, batch_idx)
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
        # TODO: call post processing handler
        # result = toolz.valmap(np.vstack, result)
        # result and original input data should be available in the post processing handler
        # output = self.postprocess(toolz.merge(data, batch))
        log.info(f"batch keys: {batch.keys()} and data keys: {data.keys()}")
        output = self.postprocess(batch)

        return output
