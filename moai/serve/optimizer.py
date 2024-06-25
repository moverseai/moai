import functools
import logging
import time
import typing

import benedict
import toolz
import torch

# from moai.serve.model import ModelServer
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
from moai.engine.runner import LightningRunner
from moai.utils.funcs import get_list

log = logging.getLogger(__name__)  # NOTE: check name when logging from serve

__all__ = ["OptimizerServer"]

# import debugpy

# debugpy.listen(("0.0.0.0", 6789))
# debugpy.wait_for_client()


class OptimizerServer(ModelServer):
    def __init__(self) -> None:
        super().__init__()
        self.handle = self.fitting_handle

    def fitting_handle(
        self, data: typing.Mapping[str, typing.Any], context: typing.Any
    ):
        """
        Handler responsible for returning a streaming response.
        This handler is used when the model is in fit mode.
        """
        log.info("Fitting handler called.")
        start_time = time.time()
        output = None  # DEBUG

        self.context = context
        metrics = self.context.metrics

        batch = self.preprocess(data)
        batch_idx = batch.get("batch_idx", 0)
        # TODO call main inference function
        # set model to training true before calling the training step
        self.model.train()
        # TODO: add post processing handler
        # Do I need to setup a lightning trainer?
        # TODO: device should be taken from the input request
        trainer = LightningRunner(
            model_callbacks=ModelCallbacks([self.model]), devices=[0]
        )

        # reimplement training step to get access to intermediate outputs (e.g. from intermediate iterations; this should be reported to the response)
        def closure(tensors, index, steps, stage, optimizer, objective):
            # def backward_fn(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
            # call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
            for step in steps:
                tensors = self.model.named_flows[step](tensors)
            self.model.named_objectives[objective](tensors)
            loss = tensors[f"{C._MOAI_LOSSES_}.total"]
            is_first_batch_to_accumulate = index % trainer.accumulate_grad_batches == 0
            if trainer.accumulate_grad_batches == 1 or not is_first_batch_to_accumulate:
                call._call_callback_hooks(trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(
                    trainer, "on_before_zero_grad", optimizer
                )
                call._call_lightning_module_hook(
                    self.trainer,
                    "optimizer_zero_grad",
                    trainer.current_epoch,
                    index,
                    optimizer,
                )
            call._call_strategy_hook(trainer, "backward", loss, optimizer)
            self.optimization_step += 1
            if monitor := toolz.get_in(
                [C._FIT_, C._OPTIMIZATION_STEP_, stage], self.model.monitor
            ):
                should_monitor = (
                    self.model.optimization_step % monitor.get("_frequency_", 1) == 0
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
                    not trainer.strategy.handles_gradient_accumulation
                    and trainer.fit_loop._should_accumulate()
                ):  # For gradient accumulation calculate loss (train step + train step end)
                    with _block_parallel_sync_behavior(trainer.strategy, block=True):
                        current_closure()  # automatic_optimization=True: perform ddp sync only when performing optimizer_step
                else:
                    if optimizer is not None:
                        self.model.optimizer_step(
                            trainer.current_epoch,
                            batch_idx,
                            optimizer,
                            current_closure,
                        )
                        # create intermediate response
                        # key should be parseed
                        # from self.context.request_ids.keys()
                        key = 0  # DEBUG
                        batch[key] = (
                            f"Running Stage {stage} and Iteration {iter} from the model."
                        )
                        # send_intermediate_predict_response(
                        #   batch,
                        #  self.context.request_ids,
                        #  "Intermediate response from the model.",
                        #  200,
                        #  self.context,
                        # )
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
                                # send_intermediate_predict_response(
                                #  batch,
                                # self.context.request_ids,
                                # "Intermediate response from the model.",
                                # 200,
                                # self.context,
                                # )
                                # TODO: this should be returned as intermediate response in the HTTP response
                # TODO: do we need to add monitoring here?
                # We should add monitoring here and use the contect metrics to report them in GET /metrics
                # if iter_monitor_stage := toolz.get_in(
                #     [C._FIT_, C._STAGES_, stage], self.monitor
                # ):
                #     frequency = toolz.get(C._FREQUENCY_, iter_monitor_stage, 1)
                #     should_monitor = iter % frequency == 0
                #     if (
                #         iter_tensor_monitor := iter_monitor_stage.get(
                #             C._MONITORS_, None
                #         )
                #     ) and should_monitor:
                #         for step in (
                #             toolz.get(C._FLOWS_, iter_monitor_stage, None) or []
                #         ):
                #             self.model.named_flows[step](batch)
                #         for metric in (
                #             toolz.get(C._METRICS_, iter_monitor_stage, None) or []
                #         ):
                #             self.model.named_metrics[metric](batch)
                #         extras = {  # TODO: step => 'lightning_step'
                #             "lightning_step": self.model.global_step,
                #             "epoch": self.model.current_epoch,
                #             "batch_idx": batch_idx,
                #             "stage": stage,
                #             "iteration": iter,
                #         }
                #         for step in iter_tensor_monitor:
                #             self.model.named_monitors[step](batch, extras)
                #         should_stop = False
                #         for criterion in get_list(iter_monitor_stage, C._TERMINATION_):
                #             if self.model.named_criteria[criterion](batch, extras):
                #                 should_stop = True
                #                 break
                #         if should_stop:
                #             log.info(
                #                 f"Terminating {stage} @ {iter} with criterion [{criterion}] !"
                #             )
                #             break
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
        metrics.add_time(
            "FittingHandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output


if __name__ == "__main__":
    from requests.structures import CaseInsensitiveDict
    from ts.context import Context, RequestProcessor

    manifest = {
        "createdOn": "06/06/2021 19:42:51",
        "runtime": "python",
        "model": {
            "modelName": "projects_shape_fit",
            "handler": "model.py",
            "modelVersion": "1.0",
        },
        "archiverVersion": "0.5.3",
    }
    # intialize model
    context = Context(
        model_name="projects_shape_fit",
        model_dir="./",
        manifest=manifest,
        batch_size=1,
        gpu=0,
        mms_version="",
    )
    import os

    # Set environment variables
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["MOVERSE_SMPL_MODEL"] = (
        "C:/Users/giorg/Documents/Projects/markerless-mocap/third_party/smplx/transfer_data/body_models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    )
    os.environ["MOVERSE_BDTK_ROOT"] = "D:/repos/bdtk"
    os.environ["MOVERSE_GENDER"] = "neutral"
    os.environ["MOVERSE_SMPL_ROOT"] = (
        "C:/Users/giorg/Documents/Projects/markerless-mocap/third_party/smplx/transfer_data/body_models"
    )
    os.environ["MOVERSE_BETAS"] = "10"
    os.environ["MOVERSE_WINDOW_SIZE"] = "10"
    os.environ["MOVERSE_STRIDE"] = "9"
    os.environ["MOVERSE_POSE_W"] = "10.78"
    os.environ["MOVERSE_SHAPE_W"] = "50.0"
    os.environ["MOVERSE_LANDMARKS_W"] = "1"
    os.environ["MOVERSE_PRIOR_ROOT"] = (
        "C:/Users/giorg/Documents/Projects/markerbased-mocap/actions/train/2024-04-12/00-14-20-svae/svae/version_0/checkpoints/epoch_5.ckpt"
    )
    os.environ["MOVERSE_PQ_ROOT"] = (
        "D:/repos/dot/actions/fit/2024-06-18/11-20-16-bundle/bundle.pq"
    )
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"

    rp = RequestProcessor(headers)
    # context.request_processor = [rp]
    context.request_processor = {
        "idx": rp,
    }

    model = ModelServer()
    context.request_ids = {
        "idx": "1fsdf",
    }
    model.initialize(context)
    model.handle(
        {
            "vrs": "//NAS5DD7C0/mov-dev/mvrs_files/lite-full-heavy-recordings/5dc19982-a7b2-4955-ad93-e162b992f22e/pushups-low-lite.mvrs",
            "body": "test",
        },
        context,
    )
    # model.handle()
