import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning_fabric
import pytorch_lightning as L
import torch
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.loggers import Logger
from lightning_fabric.strategies import Strategy
from lightning_fabric.wrappers import _unwrap_objects
from pytorch_lightning.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm
import toolz

import hydra
import hydra.utils as hyu
import omegaconf.omegaconf


import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["LightningFitter"]


class PerBatch(torch.nn.Identity, L.Callback):
    def __init__(self):
        super(PerBatch, self).__init__()

    
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: typing.Dict[
            str,
            typing.Union[
                torch.Tensor,
                typing.Sequence[torch.Tensor],
                typing.Dict[str, torch.Tensor],
            ],
        ],
        batch_idx: int,
        unused: typing.Optional[int] = 0,
    ) -> None:
        """Called when the train batch begins."""
        pl_module.initialize_parameters() if not trainer.init_once or batch_idx == 0 else None
        if batch_idx > 0:
            log.info(f"Optimizing batch {batch_idx}")
            #TODO: the folloining line is not valid for cuda accelerator
            # trainer.accelerator.setup_optimizers(trainer)
        if "inference" in pl_module.mode:
            with torch.no_grad():
                pl_module.preprocess(batch)
                pl_module(batch)
                pl_module.initialize(
                    batch
                ) if not trainer.init_once or batch_idx == 0 else None
            if pl_module.optimized_predictions:
                for key, values in pl_module.optimized_predictions.items():
                    for optim in filter(
                        lambda o: o.name == key, trainer.accelerator.optimizers
                    ):
                        for v in values:
                            params = pl_module.predictions[v]
                            if isinstance(optim, torch.optim.LBFGS) or isinstance(
                                optim, miLBFGS
                            ):
                                optim._params.append(params)
                            else:
                                optim.param_groups.append({"params": params})
        pl_module.optimization_step = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: typing.Dict[
            str,
            typing.Union[
                torch.Tensor,
                typing.Sequence[torch.Tensor],
                typing.Dict[str, torch.Tensor],
            ],
        ],
        batch: typing.Dict[
            str,
            typing.Union[
                torch.Tensor,
                typing.Sequence[torch.Tensor],
                typing.Dict[str, torch.Tensor],
            ],
        ],
        batch_idx: int,
        unused: typing.Optional[int] = 0,
    ) -> None:
        """Called when the train batch ends."""
        metrics = pl_module.validation(batch)
        pl_module.log_dict(
            metrics,
            prog_bar=True,
            logger=False,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )
        log_metrics = toolz.keymap(lambda k: f"val_{k}", metrics)
        # pl_module.log_dict(
        #     log_metrics,
        #     prog_bar=False,
        #     logger=True,
        #     on_epoch=False,
        #     on_step=True,
        #     sync_dist=True,
        # )
        # NOTE Why metrics results are not available to visualizers?
        pl_module.visualization(
            toolz.merge(batch, metrics), pl_module.optimization_step
        )
        pl_module.exporter(batch, pl_module.optimization_step)
    
    # def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    #     pl_module.exporter()

class LightningFitter(L.Trainer):
    def __init__(
        self,
        accelerator: typing.Optional[
            typing.Union[str, L.accelerators.Accelerator]
        ] = "auto",
       strategy: typing.Optional[
            typing.Union[str, L.strategies.Strategy]
        ] = "auto",
        devices: typing.Optional[typing.Union[typing.List[int], str, int]] = "auto",
        num_nodes: int = 1,
        precision: Union[str, int] = "32-true",
        logging: omegaconf.DictConfig = None,
        regularization: omegaconf.DictConfig = None,
        callbacks: omegaconf.DictConfig = None,
        model_callbacks: typing.Sequence[L.Callback] = None,
        default_root_dir: typing.Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",  # NOTE: @PTL1.5
        process_position: int = 0,
        num_processes: int = 1,
        gpus: typing.Optional[typing.Union[typing.List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: typing.Optional[typing.Union[typing.List[int], str, int]] = None,
        log_gpu_memory: typing.Optional[str] = None,
        progress_bar_refresh_rate: int = 1,
        overfit_batches: float = 0.0,
        track_grad_norm: int = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: bool = False,
        accumulate_grad_batches: typing.Union[
            int, typing.Dict[int, int], typing.List[list]
        ] = 1,
        max_epochs: int = 1000,
        min_epochs: int = 1,
        max_steps: typing.Optional[int] = -1,
        min_steps: typing.Optional[int] = None,
        limit_train_batches: typing.Union[int, float] = 1.0,
        limit_val_batches: typing.Union[int, float] = 1.0,
        limit_test_batches: typing.Union[int, float] = 1.0,
        limit_predict_batches: typing.Union[int, float] = 1.0,  # NOTE: @PTL1.5
        val_check_interval: typing.Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 10,
        sync_batchnorm: bool = False,
        enable_model_summary: bool = True,
        weights_summary: typing.Optional[str] = "full",
        weights_save_path: typing.Optional[str] = None,
        num_sanity_val_steps: int = 2,
        # NOTE: @PTL1.5 truncated_bptt_steps:       typing.Optional[int]=None,
        resume_from_checkpoint: typing.Optional[str] = None,
        profiler: typing.Optional[
        typing.Union[L.profilers.Profiler, str]
        ] = None,
        benchmark: bool = False,
        deterministic: bool = True,
        reload_dataloaders_every_epoch: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: typing.Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,  # NOTE: @PTL1.5
        auto_scale_batch_size: typing.Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: typing.Optional[list] = None,
        amp_backend: str = "native",
        amp_level: str = "O2",
        move_metrics_to_cpu: bool = False,  # NOTE: @PTL1.5
        terminate_on_nan: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        stochastic_weight_avg: bool = False,
        grad_accum_steps: int = 1,
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        relative_tolerance: float = 1e-12, # TODO: needed for manual gradient clipping?
        gradient_tolerance: float = 1e-12,
        init_once: bool = False,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """

        pytl_callbacks = (
            [hyu.instantiate(c) for c in callbacks.values()]
            if callbacks is not None
            else []
        )
        pytl_callbacks.append(PerBatch())
        if regularization is not None:
            pytl_callbacks.append(hyu.instantiate(regularization))
        # if model_callbacks:
        #     pytl_callbacks.extend(model_callbacks)
        #NOTE: Do we need to include such information here as soon as we employ the fabric?
        super(LightningFitter, self).__init__(
            accelerator=accelerator,  # Union[str, Accelerator] = "auto",
            strategy=strategy,  # Union[str, Strategy] = "auto",
            # devices=gpus,  # Union[List[int], str, int] = "auto", 
            num_nodes=num_nodes,  # int = 1,
            precision=precision,
            logger=logging,
            # logger=[hyu.instantiate(logger) for logger in logging["loggers"].values()],
            # callbacks=callbacks,
            # callbacks=pytl_callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=None,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            # enable_checkpointing=checkpoint_callback,  # PL 2.0
            # enable_progress_bar: Optional[bool] = None, # PL 2.0
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            # inference_mode: bool = True, # PL 2.0
            # use_distributed_sampler: bool = True, # PL 2.0
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            # barebones: bool = False, # PL 2.0
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=None if not default_root_dir else default_root_dir,
        )

        self.init_once = init_once

        #TODO: test if default callbacks can be bassed to fabric
        # pytl_callbacks.extend(self.callbacks)
        pytl_callbacks.extend(model_callbacks)

        self.fabric = lightning_fabric.fabric.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=1,
            precision=precision,
            plugins=plugins,
            callbacks=pytl_callbacks,
            loggers=logging,
        )
        # self.global_step = 0 TODO: do I need this here?
        self.grad_accum_steps: int = grad_accum_steps
        # self.current_epoch = 0 # TODO: do I need this here?

        # self.max_epochs = max_epochs
        # self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        # if not isinstance(limit_train_batches, int):
        #     assert limit_train_batches == float("inf")

        # if not isinstance(limit_val_batches, int):
        #     assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        # self.checkpoint_dir = checkpoint_dir #NOTE: not needed for fitter
        # self.checkpoint_frequency = checkpoint_frequency #NOTE: not needed for fitter
        self.gradient_tolerance = gradient_tolerance
        self.relative_tolerance = relative_tolerance

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
    
    def fit(
        self,
        model: L.LightningModule, # should be the optimizer model
        # train_loader: torch.utils.data.DataLoader,
        # val_loader: torch.utils.data.DataLoader,
        # ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        self.fabric.launch()

        # setup dataloaders
        # TODO: DO I need to use fabric for setting up dataloaders?
        train_loader = self.fabric.setup_dataloaders(model.train_dataloader(), use_distributed_sampler=self.use_distributed_sampler)
        # if val_loader is not None: #NOTE: not needed for fitter
            # val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        # setup model and optimizer
        if isinstance(self.fabric.strategy, lightning_fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        # optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        optimizers, schedulers = model.configure_optimizers()
        iters = list(toolz.mapcat(lambda o: o.iterations, toolz.unique(optimizers)))
        stages = list(toolz.mapcat(lambda o: o.name, toolz.unique(optimizers)))
        # assert optimizers is not None
        # optimizers should be passed to the fabric here
        model, *optimizers = self.fabric.setup(model, *optimizers)
        # NOTE: for some reason that line adds the model itself
        # as a callback to the fabric
        # remove the last callback as it is the model itself
        self.fabric._callbacks.pop()

        # optimizers should map with stages
        # stages should run for the same batch
        for batch_idx, batch in enumerate(train_loader):
            log.info(f"Optimising batch {batch_idx}")
            self.last_loss = None
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= self.limit_train_batches:
                break
            for i, (optim, iter, stage, sched) in enumerate(
                zip(optimizers, iters, stages, schedulers)
            ):
                self.fabric.call("on_train_batch_start", self, model, batch, batch_idx)
                # TODO: DO I need to initialize parameters here?
                if model.mode == "inference":
                    with torch.no_grad():
                        model.preprocess(batch)
                        model(batch)
                        #TODO: check how to handle init in that context
                        # if batch_idx == 0:
                            # init only once for the first batch
                            # self.optimizer.initialize(data)
          
                log.info(f"Optimizing stage: {stage} for {iter} iterations")
                for p in model.parameters():
                    p.requires_grad_(False)
                for pg in optim.param_groups:
                    for p in pg["params"]:
                        p.requires_grad_(True)
                
                def closure():
                    model.optimizer_zero_grad(
                        epoch=0, batch_idx=0, optimizer=optim
                    )
                    self.loss = model.training_step(
                        batch=batch, batch_idx=batch_idx, optimizer_idx=i
                    )["loss"]
                    #TODO: change to fabric default backward
                    self.loss.backward()
                    model.optimization_step += 1
                    batch["__moai__"][
                        "optimization_step"
                    ] = model.optimization_step
                    return self.loss
                
                for j in range(iter):
                    log.info(f"Optimizing stage '{stage}' iteration {j}/{iter}")
                    optim.step(closure=closure)
                    current_loss = self.loss
                    if hasattr(optim, "assign"):
                        with torch.no_grad():
                            optim.assign(batch)
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
                            f"Optimization stage '{stage}' stopped at iteration {j}/{iter}."
                        )
                        break
                    self.last_loss = current_loss
                sched.step()
                self.last_loss = None
                model.optimization_step = 0
            metrics = model.validation(batch)
            self.fabric.call("on_train_batch_end", self, model, metrics, batch, batch_idx)
        
        # call on test epoch end
        self.fabric.call("on_train_epoch_end", self, model)


        # call on train epoch end
        # for optimizer, scheduler in zip(optimizers, schedulers):
        #     model, scheduler = self.fabric.setup(model, optimizer)

        
        #     # model, optimizer = self.fabric.setup(model, optimizer)

        #     # assemble state (current epoch and global step will be added in save)
        #     # state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        #     while not self.should_stop:
        #         #TODO: add scheduling logic to the train loop
        #         self.train_loop(
        #             model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=None
        #         )

        #         #NOTE: validation not needed for fitter
        #         # if self.should_validate:
        #         #     self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

        #         #TODO: add scheduling logic here
        #         # self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

        #         self.current_epoch += 1

        #         # stopping condition on epoch level
        #         if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
        #             self.should_stop = True

        #         # self.save(state)

        #     # reset for next fit call
        #     self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[lightning_fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        # self.fabric.call("on_train_epoch_start")
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            #TODO: Callables are not working
            # self.fabric.call("on_train_batch_start", batch, batch_idx)
            #TODO: workaround for now as we need to understand why fabric appends an additional callback
            if len(self.fabric._callbacks) > 1:
                # pop the last callback
                self.fabric._callbacks.pop()
            self.fabric.call("on_train_batch_start", self, model, batch, batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", self, model, optimizer)

                # optimizer step runs train step internally through closure
                optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))
                self.fabric.call("on_before_zero_grad", self, model, optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.fabric.call("on_train_batch_end", self, model, [] , batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            # self._format_iterable(iterable, self._current_train_return, "train")

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model)):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(self, model: L.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx)

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", self, model, loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward", self, model)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss

    def run(self, model: L.LightningModule):
        """Runs the training loop.

        Args:
            model: the LightningModule to train

        """
        self.fit(model)

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[lightning_fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[lightning_fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[lightning_fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, lightning_fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, lightning_fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, lightning_fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (lightning_fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)


if __name__ == "__main__":
    # do some testing here
    fitter = LightningFitter()
    print("Fitter instantiated.")
    # run fitter
