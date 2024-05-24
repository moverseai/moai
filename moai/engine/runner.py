from moai.core.execution.constants import Constants

import pytorch_lightning as L
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import tablib
import torch
import toolz
import logging
from collections import defaultdict
import tablib
import numpy as np
import os
from moai.log.lightning.loggers.tabular import Tabular
import numpy as np
import os

from torchmetrics import Metric as TorchMetric
from moai.validation.metrics.generation.fid import FID
from moai.validation.metrics.generation.diversity import Diversity

log = logging.getLogger(__name__)

__all__ = ["LightningRunner"]

class BatchMonitor(L.Callback):

    def setup(self, trainer: L.Trainer, module: L.LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        # module.initialize_parameters()
        module.setup_initializers()

    @torch.no_grad
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        module: L.LightningModule,
        batch: typing.Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        module.optimization_step = 0
        # call initialize per batch
        module.batch_initializers()
    
    def on_train_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when the train epoch begins."""
        module.epoch_initializers()

    @torch.no_grad
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        module: L.LightningModule,
        outputs: L.utilities.types.STEP_OUTPUT,
        batch: typing.Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends.
        Note: The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        # for k, monitor_batch in module.monitor.get('fit', {}).get('batch', {}).items():
        monitor_batch = toolz.get_in(["fit", "batch"], module.monitor)
        if monitor_batch is not None:
            for _, monitor_named_batch in monitor_batch.items():
                is_now = batch_idx % monitor_named_batch['frequency'] == 0
                if not is_now:
                    return # continue
                #NOTE: should detach
                for step in monitor_named_batch.get('steps', []):
                    outputs = module.graphs[step](outputs)
                for monitor_metrics in monitor_named_batch.get('metrics', []):
                    module.named_metrics[monitor_metrics](outputs)
                extras = {
                    'step': module.global_step, 'epoch': module.current_epoch,
                    'batch_idx': batch_idx,
                }
                for monitor_tensors in monitor_named_batch.get('tensors', []):
                    module.named_monitors[monitor_tensors](outputs, extras)
                if 'losses' in outputs:
                    losses = toolz.merge(outputs['losses']['weighted'], {
                        'total': outputs['losses']['total'],                
                    })
                    module.log_dict(losses, prog_bar=True, logger=False, on_step=True, on_epoch=False)
                    losses = toolz.keymap(lambda k: f'train/loss/{k}', losses)
                    losses['epoch'] = int(trainer.current_epoch)
                    module.log_dict(losses, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                if Constants._MOAI_METRICS_ in outputs:
                    flattened_metrics = outputs[Constants._MOAI_METRICS_].flatten(separator='/')
                    module.log_dict(flattened_metrics, prog_bar=True, logger=False, on_step=True, on_epoch=False)
                    metrics = toolz.keymap(lambda k: f'train/metric/{k}',  flattened_metrics)
                    # metrics = outputs[Constants._MOAI_METRICS_].flatten(separator='/')
                    # metrics['epoch'] = trainer.current_epoch
                    module.log_dict(metrics, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        # return outputs
    
    @torch.no_grad
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        module: L.LightningModule,
        outputs: L.utilities.types.STEP_OUTPUT,
        batch: typing.Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        datasets = list(module.data.test.iterator.datasets.keys())
        # TODO: move batch tensors to cpu
        if "metrics" in batch:
            # metrics = toolz.keymap(lambda k: f"test/metric/{k}", batch["metrics"])
            metrics = toolz.keymap(lambda k: f"test/metric/{k}/{datasets[dataloader_idx]}", batch['metrics'])
            # move metrics to cpu numpy before logging
            # log_metrics = toolz.keymap(lambda k: f"test_{k}/{list(self.data.test.iterator.datasets.keys())[dataloader_idx]}", metrics)
            module.log_dict(
                metrics, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
            module.test_step_outputs[datasets[dataloader_idx]].append(metrics)
    
    @torch.no_grad
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        #NOTE: do we need to calculate the mean of the metrics and saved them as a separate file?
        all_metrics = defaultdict(list)
        log_metrics = defaultdict(list)
        outputs = pl_module.test_step_outputs
        for i, dataset in enumerate(outputs):
            o = outputs[dataset]
            keys = next(iter(o), { }).keys()
            metrics = { }
            for key in keys:
                metrics[key] = np.mean(np.array(
                    [d[key].item() for d in o if key in d]
                ))
                all_metrics[key].append(metrics[key])
            log_metrics[dataset] = metrics
        #NOTE: logging a dict is not supported in PTL 2.0
        # if there is a tabular logger write the metrics to a csv file
        logger = pl_module.logger
        if not isinstance(logger, Tabular):
            return
        for dataset in log_metrics.keys():
            ds = tablib.Dataset([v for v in log_metrics[dataset].values()], headers=toolz.keymap(lambda k: k.split("/")[2], dict(log_metrics[dataset])).keys())
            with open(os.path.join(os.getcwd(), f'{logger.name}_{dataset}_test_average.csv'), 'a', newline='') as f:
                 f.write(ds.export('csv'))
        pl_module.test_step_outputs.clear()

    @torch.no_grad
    def on_validation_batch_end(self,
        trainer: L.Trainer, module: L.LightningModule,
        outputs: L.utilities.types.STEP_OUTPUT,
        batch: typing.Mapping[str, torch.Tensor], batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not hasattr(module.data, 'val'):
            log.warning("No validation data found, validation step will be omitted.")
            return
        datasets = list(module.data.val.iterator.datasets.keys())
        if Constants._MOAI_METRICS_ in outputs:
            flattened_metrics = outputs[Constants._MOAI_METRICS_].flatten(separator='/')
            scalar_metrics = toolz.keymap( # format: val/metric/named_metric/dataset
                lambda k: f'val/metric/{k}/{datasets[dataloader_idx]}', 
                toolz.valfilter(
                    # lambda x: len(x.shape) == 0, 
                    lambda x: torch.numel(x) <= 1, flattened_metrics                    
                )                
            )
            dataset_val_metrics = toolz.valmap(
                lambda v: toolz.keymap(# extract metric_type | metric_name
                    lambda k: k.split("/")[2], 
                    dict(v)
                ),
                toolz.groupby( # group by last key, i.e. dataset
                    lambda k: k[0].split('/')[-1],
                    scalar_metrics.items()
                )
            )
            if scalar_metrics:
                module.scalar_metrics[datasets[dataloader_idx]].append( # format: dataset/named_metric
                    toolz.valmap(
                        lambda x: x.detach().cpu().numpy(),
                        dataset_val_metrics[datasets[dataloader_idx]]
                    )
                )
                scalar_metrics = toolz.valfilter(#NOTE: filter empty before log
                    lambda x: torch.numel(x) == 1, scalar_metrics
                )
                scalar_metrics["epoch"] = module.current_epoch                
                module.log_dict(scalar_metrics, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            non_scalar_metrics = toolz.keymap(
                lambda k: f'{datasets[dataloader_idx]}/{k}', 
                toolz.valfilter(
                    # lambda x: len(x.shape) > 0,
                    # outputs[Constants._MOAI_METRICS_].flatten(separator='/')
                    lambda x: torch.numel(x) > 1, flattened_metrics                    
                )
            )
            if non_scalar_metrics:
                module.non_scalar_metrics[datasets[dataloader_idx]].append(
                    toolz.valmap(
                        lambda x: x.detach().cpu().numpy(),
                        toolz.keymap(
                            lambda k: k.split("/")[1],
                            non_scalar_metrics
                        )
                    )
                )

    @torch.no_grad
    def on_validation_epoch_end(self, 
        trainer: L.Trainer, 
        module: L.LightningModule
    ) -> None:
        all_scalar_metrics = {}
        all_non_scalar_metrics = {}
        all_features = {}
        all_metrics = {}
        log_all_metrics = {} # defaultdict(list)
        if module.scalar_metrics:
            for i, (dataset, metrics) in enumerate(module.scalar_metrics.items()):
                all_scalar_metrics[dataset] = {}
                all_scalar_metrics[dataset]['epoch'] = int(module.current_epoch)
                keys = next(iter(metrics), { }).keys()
                for metric_name in keys:
                    # metric_type, metric_name = key.split("|")
                    # full_name, metric_module = toolz.first(toolz.filter(
                    #     lambda n: n[0].endswith('multiclass_acc'), 
                    #     module.named_metrics.named_modules()
                    # ))
                    metric_module = module.metric_name_to_module[metric_name]
                    if isinstance(metric_module, TorchMetric):                        
                        all_scalar_metrics[dataset][metric_name] = \
                            metric_module.compute().detach().cpu().numpy()
                        metric_module.reset()
                    else:
                        all_scalar_metrics[dataset][metric_name] = \
                            metric_module.compute(d[metric_name] for d in metrics)
                    log_all_metrics[metric_name] = float(all_scalar_metrics[dataset][metric_name])
            module.scalar_metrics.clear()  # free memory
        
        if module.non_scalar_metrics:
            for i, dataset in enumerate(module.non_scalar_metrics):
                all_non_scalar_metrics[dataset] = {}
                o = module.non_scalar_metrics[dataset]
                keys = next(iter(o), { }).keys()
                features = {}
                for key in keys:
                    features[key] = np.vstack(
                            [d[key] for d in o if key in d]
                    )
                all_features[dataset] = features
                fid_metric = toolz.get_in(['fid'], module.generation_metrics) or {}
                if fid_metric:
                    dict_elem = toolz.get_in(['pred'], fid_metric)
                    for elem in range(len(dict_elem)):
                        all_non_scalar_metrics[dataset][f"{fid_metric['out'][elem]}"] = FID().forward(
                                pred=torch.from_numpy(all_features[dataset][fid_metric['pred'][elem]]),
                                gt=torch.from_numpy(all_features[dataset][fid_metric['gt'][elem]])
                        ).item()
                        log_all_metrics[f"{fid_metric['out'][elem]}"].append(all_non_scalar_metrics[dataset][f"{fid_metric['out'][elem]}"])
                div_metric = toolz.get_in(['diversity'], module.generation_metrics) or {}
                if div_metric:
                    dict_elem = toolz.get_in(['pred'], div_metric)
                    for elem in range(len(dict_elem)):
                        all_non_scalar_metrics[dataset][f"{div_metric['out'][elem]}"] = Diversity().forward(
                                    pred=torch.from_numpy(all_features[dataset][div_metric['pred'][elem]]),
                            ).item()
                        log_all_metrics[f"{div_metric['out'][elem]}"].append(all_non_scalar_metrics[dataset][f"{div_metric['out'][elem]}"])
                all_metrics[dataset] = toolz.merge(all_scalar_metrics[dataset], all_non_scalar_metrics[dataset])\
                                                        if len(all_scalar_metrics.keys()) > 0\
                                                        else all_non_scalar_metrics[dataset]
            module.non_scalar_metrics.clear()
        for dataset in all_metrics.keys():
            ds = tablib.Dataset(headers=all_metrics[dataset].keys()) \
                if module.current_epoch < 1 \
                else tablib.Dataset(headers=False)
            ds.append([v for v in all_metrics[dataset].values()])
            with open(os.path.join(os.getcwd(), f'{module.logger.name}_{dataset}_val_average.csv'), 'a', newline='') as f:
                 f.write(ds.export('csv'))
        # log_all_metrics = toolz.valmap(lambda v: sum(v) / len(v), log_all_metrics)
        module.log_dict(log_all_metrics, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
        log_all_metrics = toolz.keymap(lambda k: f"val/metric/{k}", log_all_metrics)
        module.log_dict(log_all_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)

# class PerBatch(torch.nn.Identity, L.Callback):
#     def __init__(self):f
#         super(PerBatch, self).__init__()

#     def on_train_batch_start(
#         self,
#         trainer: L.Trainer,
#         pl_module: L.LightningModule,
#         batch: typing.Dict[
#             str,
#             typing.Union[
#                 torch.Tensor,
#                 typing.Sequence[torch.Tensor],
#                 typing.Dict[str, torch.Tensor],
#             ],
#         ],
#         batch_idx: int,
#         unused: typing.Optional[int] = 0,
#     ) -> None:
#         """Called when the train batch begins."""
#         pl_module.initialize_parameters() if not trainer.init_once or batch_idx == 0 else None
#         if batch_idx > 0:
#             trainer.accelerator.setup_optimizers(trainer)
#         if "inference" in pl_module.mode:
#             with torch.no_grad():
#                 pl_module.preprocess(batch)
#                 pl_module(batch)
#                 pl_module.initialize(
#                     batch
#                 ) if not trainer.init_once or batch_idx == 0 else None
#             if pl_module.optimized_predictions:
#                 for key, values in pl_module.optimized_predictions.items():
#                     for optim in filter(
#                         lambda o: o.name == key, trainer.accelerator.optimizers
#                     ):
#                         for v in values:
#                             params = pl_module.predictions[v]
#                             if isinstance(optim, torch.optim.LBFGS) or isinstance(
#                                 optim, miLBFGS
#                             ):
#                                 optim._params.append(params)
#                             else:
#                                 optim.param_groups.append({"params": params})
#         pl_module.optimization_step = 0

#     def on_train_batch_end(
#         self,
#         trainer: L.Trainer,
#         pl_module: L.LightningModule,
#         outputs: typing.Dict[
#             str,
#             typing.Union[
#                 torch.Tensor,
#                 typing.Sequence[torch.Tensor],
#                 typing.Dict[str, torch.Tensor],
#             ],
#         ],
#         batch: typing.Dict[
#             str,
#             typing.Union[
#                 torch.Tensor,
#                 typing.Sequence[torch.Tensor],
#                 typing.Dict[str, torch.Tensor],
#             ],
#         ],
#         batch_idx: int,
#         unused: typing.Optional[int] = 0,
#     ) -> None:
#         """Called when the train batch ends."""
#         metrics = pl_module.validation(batch)
#         pl_module.log_dict(
#             metrics,
#             prog_bar=True,
#             logger=False,
#             on_epoch=False,
#             on_step=True,
#             sync_dist=True,
#         )
#         log_metrics = toolz.keymap(lambda k: f"val_{k}", metrics)
#         pl_module.log_dict(
#             log_metrics,
#             prog_bar=False,
#             logger=True,
#             on_epoch=False,
#             on_step=True,
#             sync_dist=True,
#         )
#         # NOTE Why metrics results are not available to visualizers?
#         pl_module.visualization(
#             toolz.merge(batch, metrics), pl_module.optimization_step
#         )
#         pl_module.exporter(batch, pl_module.optimization_step)


# class OptimizationLoop(pytorch_lightning.loops.OptimizerLoop):
#     def __init__(
#         self,
#         relative_tolerance: float = 1e-9,
#         gradient_tolerance: float = 1e-9,
#     ):
#         super(OptimizationLoop, self).__init__()
#         self.last_loss = None
#         self.gradient_tolerance = gradient_tolerance
#         self.relative_tolerance = relative_tolerance

#     def relative_check(self, prev: torch.Tensor, current: torch.Tensor) -> float:
#         relative_change = (prev - current) / max([prev.abs(), current.abs(), 1.0])
#         return relative_change <= self.relative_tolerance

#     def gradient_check(
#         self, param_groups: typing.Sequence[typing.Dict[str, torch.nn.Parameter]]
#     ) -> bool:
#         return all(
#             p.grad.view(-1).max().abs().item() < self.gradient_tolerance
#             for p in toolz.concat((g["params"] for g in param_groups))
#             if p.grad is not None
#         )

#     def is_any_param_nan(self, optimizer: torch.optim.Optimizer) -> bool:
#         for pg in optimizer.param_groups:
#             for p in pg["params"]:
#                 if not torch.all(torch.isfinite(p)):
#                     return True
#         return

#     def advance(
#         self, batch: typing.Any, *args: typing.Any, **kwargs: typing.Any
#     ) -> None:
#         # optimizers = toolz.mapcat(
#         #     lambda io: ((io[0], io[1], it, n) for (it, n) in zip(
#         #         io[1].iterations, io[1].name)
#         #     ),
#         #     enumerate(self._optimizers)
#         # )
#         iters = list(
#             toolz.mapcat(lambda o: o.iterations, toolz.unique(self._optimizers))
#         )
#         stages = list(toolz.mapcat(lambda o: o.name, toolz.unique(self._optimizers)))
#         # for i, optim in tqdm.tqdm(enumerate(self._optimizers), desc=f"Optimization"):
#         for i, (optim, iters, stage) in tqdm.tqdm(
#             enumerate(zip(self._optimizers, iters, stages)), desc=f"Optimization"
#         ):
#             # NOTE: probably not required as lightning already falses out all params
#             #           and toggles only the state of those in the current optimizer
#             for p in self.trainer.lightning_module.parameters():
#                 p.requires_grad_(False)
#             for pg in optim.param_groups:
#                 for p in pg["params"]:
#                     p.requires_grad_(True)
#             # for _ in tqdm.tqdm(range(optim.iterations), desc=f"Stage: {optim.name}"):
#             for j in tqdm.tqdm(range(iters), desc=f"Stage: {stage}"):
#                 super(OptimizationLoop, self).advance(batch, args, kwargs)
#                 self.optim_progress.optimizer_position = i
#                 current_loss = self._outputs[i]["loss"]
#                 if hasattr(optim, "assign"):
#                     with torch.no_grad():
#                         optim.assign(batch)
#                 if (
#                     (
#                         self.last_loss is not None
#                         and self.relative_check(self.last_loss, current_loss)
#                     )
#                     or self.gradient_check(self._optimizers[i].param_groups)
#                     or not torch.isfinite(current_loss)
#                     or self.is_any_param_nan(optim)
#                 ):
#                     log.warning(
#                         f"Optimization stage '{stage}' stopped at iteration {j}/{iters}."
#                     )
#                     break
#                 self.last_loss = current_loss
#             self.last_loss = None
#             self.optim_progress.optimizer_position = i + 1
#         self.optim_progress.optimizer_position = i + 1


class LightningRunner(L.Trainer):
    def __init__(self,
        # logging: omegaconf.DictConfig = None,
        loggers: omegaconf.DictConfig = None,
        checkpoint: omegaconf.DictConfig = None,
        regularization: omegaconf.DictConfig = None,
        callbacks: omegaconf.DictConfig = None,
        model_callbacks: typing.Sequence[L.Callback] = None,
        default_root_dir: typing.Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",  # NOTE: @PTL1.5
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        devices: typing.Optional[typing.Union[typing.List[int], str, int]] = "auto",
        gpus: typing.Optional[typing.Union[typing.List[int], str, int]] = None,
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
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
        max_epochs: int = 1,
        # min_epochs:                 int=1,
        max_steps: typing.Optional[int] = None,
        min_steps: typing.Optional[int] = None,
        limit_train_batches: typing.Union[int, float] = 1.0,
        limit_val_batches: typing.Union[int, float] = 1.0,
        limit_test_batches: typing.Union[int, float] = 1.0,
        limit_predict_batches: typing.Union[int, float] = 1.0,  # NOTE: @PTL1.5
        val_check_interval: typing.Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 10,
        accelerator: typing.Optional[
            typing.Union[str, L.accelerators.Accelerator]
        ] = "auto",
        strategy: str = "auto",
        sync_batchnorm: bool = False,
        precision: int = 32,
        enable_model_summary: bool = True,
        weights_summary: typing.Optional[str] = "full",
        weights_save_path: typing.Optional[str] = None,
        num_sanity_val_steps: int = 2,
        # NOTE: @PTL1.5 truncated_bptt_steps:       typing.Optional[int]=None,
        resume_from_checkpoint: typing.Optional[str] = None,
        profiler: typing.Optional[typing.Union[L.profilers.Profiler, bool, str]] = None,
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
        # NOTE: @PTL1.5 distributed_backend:        typing.Optional[str]=None,
        # NOTE: @PTL1.5 automatic_optimization:     bool=True,
        loops: omegaconf.DictConfig = None,
        relative_tolerance: float = 1e-9,
        gradient_tolerance: float = 1e-9,
        **kwargs,
    ):
        # if (
        #     logging and "_target_" not in logging
        # ):  # TODO: needs a workaround for other viz types (e.g. not visdom) if they are moved to top level
        #     # TODO: wrap config field setting into a helper method
        #     omegaconf.OmegaConf.set_struct(logging, False)
        #     logging["_target_"] = "moai.log.lightning.Collection"
        #     omegaconf.OmegaConf.set_struct(logging, True)
        # logger = hyu.instantiate(logging) if logging is not None else milog.NoOp()
        loggers = [hyu.instantiate(logger) for logger in loggers.values()]
        # if logging and logging.loggers:            
            # loggers += [hyu.instantiate(logger) for logger in logging.loggers.values()]

        # if loops is None:
        #     pytl_callbacks = [PerBatch()]
        # elif loops.callbacks is None:
        #     pytl_callbacks = [PerBatch()]
        # else:
        #     pytl_callbacks = [hyu.instantiate(loops.callbacks)]
        pytl_callbacks = [BatchMonitor()] #TODO: only when moai model is used, should not be used for custom models
        # pytl_callbacks = [PerBatch() if loops is None or loops.callbacks in None else hyu.instantiate(loops.callbacks)]
        pytl_callbacks.extend(
            [hyu.instantiate(c) for c in callbacks.values()]
            if callbacks is not None
            else []
        )
        checkpoint_callback = False
        if checkpoint is not None:
            pytl_callbacks.append(hyu.instantiate(checkpoint))
            checkpoint_callback = True
        if regularization is not None:
            pytl_callbacks.append(hyu.instantiate(regularization))
        if model_callbacks:
            pytl_callbacks.extend(model_callbacks)
        self.init_once = (
            kwargs.pop("init_once") if "init_once" in kwargs else False
        )  # default should be False
        # TODO: add optimizer reset callback https://github.com/PyTorchLightning/pytorch-lightning/issues/3095
        # TODO: add weight re-init callback
        # TODO: add inference callback with no grad forward
        super().__init__(
            logger=loggers,  # logger,
            # checkpoint_callback=checkpoint_callback,
            callbacks=pytl_callbacks,
            default_root_dir=None if not default_root_dir else default_root_dir,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,  # NOTE: @PTL1.5
            # process_position=process_position,
            num_nodes=num_nodes,
            devices=devices,  # NOTE @PTL1.5 #TODO: check if needed
            # gpus=gpus,
            # auto_select_gpus=auto_select_gpus,
            # tpu_cores=tpu_cores,
            # log_gpu_memory=log_gpu_memory,
            # progress_bar_refresh_rate=progress_bar_refresh_rate,
            overfit_batches=overfit_batches,
            # track_grad_norm=track_grad_norm,
            check_val_every_n_epoch=check_val_every_n_epoch,
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            min_epochs=1,
            max_steps=-1,
            min_steps=-1,
            max_time=None,  # NOTE @PTL1.5 #TODO: check if needed
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,  # NOTE: @PTL1.5
            val_check_interval=val_check_interval,
            # flush_logs_every_n_steps=flush_logs_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            strategy=strategy,  # NOTE: @PTL1.5
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            enable_model_summary=enable_model_summary,
            enable_progress_bar=True,
            enable_checkpointing=checkpoint_callback,
            # weights_summary=weights_summary,
            # weights_save_path=weights_save_path,
            num_sanity_val_steps=num_sanity_val_steps,
            # NOTE: @PTL1.5 truncated_bptt_steps=truncated_bptt_steps,
            # resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            # reload_dataloaders_every_epoch=reload_dataloaders_every_epoch,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            # auto_lr_find=auto_lr_find,
            detect_anomaly=detect_anomaly,  # NOTE: @PTL1.5
            # replace_sampler_ddp=replace_sampler_ddp,
            # terminate_on_nan=terminate_on_nan,
            # auto_scale_batch_size=auto_scale_batch_size,
            # prepare_data_per_node=prepare_data_per_node,
            plugins=plugins,
            # amp_backend=amp_backend,
            # NOTE: @PTL1.5 fixdistributed_backend=distributed_backend,
            # amp_level=amp_level,
            # NOTE: @PTL1.5 fixautomatic_optimization=automatic_optimization,
            # move_metrics_to_cpu=move_metrics_to_cpu,
            # multiple_trainloader_mode=multiple_trainloader_mode,
            # stochastic_weight_avg=stochastic_weight_avg,
            # num_processes=num_processes,  # NOTE: @PTL1.5 fix
            **kwargs,
        )

    def run(self, model):
        self.fit(model)
