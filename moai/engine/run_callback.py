from moai.core.execution.constants import Constants as C
from moai.log.lightning.loggers.tabular import Tabular
from collections import defaultdict
from toolz.curried import merge_with

import pytorch_lightning as L
import hydra.utils as hyu
import typing
import tablib
import torch
import toolz
import logging
import tablib
import numpy as np
import os

log = logging.getLogger(__name__)

__all__ = ["RunCallback"]

class RunCallback(L.Callback):

    def setup(self, trainer: L.Trainer, module: L.LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
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
        module.batch_initializers() # call initialize per batch
        if module.process[C._FIT_].get(C._REFRESH_OPTIMIZERS_, False):
            module.reset_optimization()
            trainer.strategy.setup_optimizers(trainer)
    
    @torch.no_grad
    def on_predict_batch_start(self, trainer: L.Trainer, module: L.LightningModule, batch: hyu.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        module.batch_initializers()

    @torch.no_grad
    def on_test_batch_start(self, trainer: L.Trainer, module: L.LightningModule, batch: hyu.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        module.batch_initializers()
    
    @torch.no_grad
    def on_train_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when the train epoch begins."""
        module.epoch_initializers()
        if module.schedule and module.current_epoch >= (scheduled_epoch := module.schedule[0][C._EPOCH_]):
            log.info(f"Updating execution @ epoch {scheduled_epoch}.")
            popped = module.schedule.popleft()
            #TODO: add remodel/modifications here as well
            # module.process[C._FIT_][C._BATCH_] = popped[C._SCHEDULE_STEP_]#NOTE: rename process?
            # module.process[C._VAL_][C._BATCH_] = popped[C._SCHEDULE_STEP_]#NOTE: rename process?
            if new_fit := popped.get(C._SCHEDULE_FIT_, None): #deep merge from https://github.com/pytoolz/toolz/issues/281
                module.process[C._FIT_] = merge_with(merge_with(toolz.merge), (module.process[C._FIT_], new_fit))
            if new_val := popped.get(C._SCHEDULE_VAL_, None):
                module.process[C._VAL_] = merge_with(merge_with(toolz.merge), (module.process[C._VAL_], new_val))
        
    @torch.no_grad
    def on_predict_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when the predict epoch begins."""
        module.epoch_initializers()
    
    @torch.no_grad
    def on_test_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
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
        if C._MOAI_LOSSES_ in outputs:
            if losses := toolz.merge(
                outputs[f"{C._MOAI_LOSSES_}.weighted"],
                {'total': outputs[f"{C._MOAI_LOSSES_}.total"]}
            ):
                module.log_dict(losses, prog_bar=True, logger=False, on_step=True, on_epoch=False)
                losses = toolz.keymap(lambda k: f'train/loss/{k}', losses)
                losses['epoch'] = int(trainer.current_epoch)
                module.log_dict(losses, prog_bar=False, logger=True, on_step=True, on_epoch=False)        
        if monitor_batch := toolz.get_in([C._FIT_, C._BATCH_], module.monitor):
            for _, monitor_named_batch in monitor_batch.items():
                is_now = batch_idx % monitor_named_batch[C._FREQUENCY_] == 0
                if not is_now:
                    continue
                #NOTE: should detach
                for step in monitor_named_batch.get(C._FLOWS_, []):
                    outputs = module.graphs[step](outputs)
                for monitor_metrics in monitor_named_batch.get(C._METRICS_, []):
                    module.named_metrics[monitor_metrics](outputs)
                extras = {
                    'lightning_step': module.global_step, 'epoch': module.current_epoch,
                    'batch_idx': batch_idx,
                }
                for monitor_tensors in monitor_named_batch.get(C._MONITORS_, []):
                    module.named_monitors[monitor_tensors](outputs, extras)
        if C._MOAI_METRICS_ in outputs:
            if flattened_metrics := outputs[C._MOAI_METRICS_].flatten(separator='/'):
                module.log_dict(flattened_metrics, prog_bar=True, logger=False, on_step=True, on_epoch=False)
                metrics = toolz.keymap(lambda k: f'train/metric/{k}',  flattened_metrics)                
                module.log_dict(metrics, prog_bar=False, logger=True, on_step=True, on_epoch=False)

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
        if "metrics" in batch:#TODO: update !! use C._MOAI_METRICS_
            metrics = toolz.keymap(lambda k: f"test/metric/{k}/{datasets[dataloader_idx]}", batch['metrics'])
            # move metrics to cpu numpy before logging
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
        if not hasattr(module.data, 'val'):#NOTE: is this necessary?
            log.warning("No validation data found, validation batch end will be omitted.")
            return
        datasets = list(module.data.val.iterator.datasets.keys())
        if C._MOAI_METRICS_ in outputs:
            flattened_metrics = outputs[C._MOAI_METRICS_].flatten(separator='/')
            scalar_metrics = toolz.keymap( # format: val/metric/named_metric/dataset
                lambda k: f'val/metric/{k}/{datasets[dataloader_idx]}', 
                toolz.valfilter(
                    lambda x: torch.numel(x) == 1, flattened_metrics 
                    # lambda x: torch.numel(x) <= 1, flattened_metrics
                )                
            )
            dataset_val_metrics = toolz.valmap(
                lambda v: toolz.keymap(
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
                scalar_metrics["epoch"] = module.current_epoch                
                module.log_dict(scalar_metrics, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            non_scalar_metrics = toolz.keymap(
                lambda k: f'{datasets[dataloader_idx]}/{k}', 
                toolz.valfilter(
                    lambda x: torch.numel(x) > 1, flattened_metrics                    
                )
            )
            if non_scalar_metrics:
                module.non_scalar_metrics[datasets[dataloader_idx]].append(
                    toolz.valmap(
                        lambda x: x.detach().cpu().numpy(),
                        toolz.keymap(
                            lambda k: f'{k.split("/")[1]}/{k.split("/")[2]}',
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
        all_metrics = {}
        log_all_metrics = {}
        if module.scalar_metrics:
            for i, (dataset, metrics) in enumerate(module.scalar_metrics.items()):
                all_scalar_metrics[dataset] = {}
                all_scalar_metrics[dataset]['epoch'] = int(module.current_epoch)
                keys = next(iter(metrics), { }).keys()
                for metric_name in keys:
                    metric_module = module.metric_name_to_module[metric_name]
                    all_scalar_metrics[dataset][metric_name] = \
                        metric_module.compute(np.stack([d[metric_name] for d in metrics]))
                    log_all_metrics[f"{metric_name}/{dataset}"] = float(all_scalar_metrics[dataset][metric_name])
                all_metrics[dataset] = all_scalar_metrics[dataset]
            module.scalar_metrics.clear()  # free memory
        
        if module.non_scalar_metrics:
            for i, (dataset, metrics) in enumerate(module.non_scalar_metrics.items()):
                all_non_scalar_metrics[dataset] = {}
                keys = next(iter(metrics), { }).keys()
                metric_names = []
                for metric_name in keys:
                    metric_name, metric_input_key = metric_name.split('/')
                    if metric_name not in metric_names:
                        metric_names.append(metric_name)
                for metric_name in metric_names:
                    metric_module = module.metric_name_to_module[metric_name]
                    inputs = list(toolz.keyfilter(lambda k: k.startswith(metric_name), metrics[0]).keys())
                    args = []
                    [args.append(np.vstack([d[i] for d in metrics])) for i in inputs]
                    #TODO: map keys to the ordered inputs of each metric
                    all_non_scalar_metrics[dataset][metric_name] = metric_module.compute(*args)
                    log_all_metrics[f"{metric_name}/{dataset}"] = float(all_non_scalar_metrics[dataset][metric_name])
                all_metrics[dataset] = toolz.merge(all_scalar_metrics[dataset], all_non_scalar_metrics[dataset])\
                    if len(all_scalar_metrics.keys()) > 0\
                    else all_non_scalar_metrics[dataset]
            module.non_scalar_metrics.clear()  # free memory
        for dataset in all_metrics.keys():
            ds = tablib.Dataset(headers=all_metrics[dataset].keys()) \
                if module.current_epoch < 1 \
                else tablib.Dataset(headers=False)
            ds.append([v for v in all_metrics[dataset].values()])
            with open(os.path.join(os.getcwd(), f'{module.logger.name}_{dataset}_val_average.csv'), 'a', newline='') as f:
                 f.write(ds.export('csv'))
        module.log_dict(log_all_metrics, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
        log_all_metrics = toolz.keymap(lambda k: f"val/metric/{k}", log_all_metrics)
        module.log_dict(log_all_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)