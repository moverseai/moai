from moai import __version__ as miV
from moai.utils.engine import NoOp as NoInterval
from moai.parameters.optimization import NoOp as NoOptimization
from moai.parameters.optimization.schedule import NoOp as NoScheduling
from moai.parameters.initialization import Default as NoInit
from moai.validation import (
    NoOp as NoValidation,
    Collection as DefaultValidation,
)
from moai.supervision import (
    NoOp as NoSupervision,
    Weighted as DefaultSupervision,
)
from moai.data.iterator import Indexed
from collections import defaultdict

import torch
import numpy as np
import pytorch_lightning
import omegaconf.omegaconf 
import hydra.utils as hyu
import typing
import toolz
import logging
import tablib
import os

log = logging.getLogger(__name__)

__all__ = ['FeedForward']

def _create_supervision_block(
    cfg: omegaconf.DictConfig,
    force: bool=True
):
    if force and not cfg:
        log.warning("Empty supervision block in feedforward model.")
    if not cfg:
        return NoSupervision()
    if '_target_' not in cfg:
        return DefaultSupervision(**cfg)
    else:
        return hyu.instantiate(cfg)

def _create_validation_block(
    cfg: omegaconf.DictConfig,
    force: bool=True
):
    if force and not cfg:
        log.warning("Empty validation block in feedforward model.")    
    if not cfg:
        return NoValidation()
    if '_target_' not in cfg:
        return DefaultValidation(**cfg)
    else:
        return hyu.instantiate(cfg)

def _create_processing_block(
    cfg: omegaconf.DictConfig, 
    attribute: str, 
    monads: omegaconf.DictConfig,
    force: bool=False
):
    if force and not cfg and attribute in cfg:
        log.warning(f"Empty processing block ({attribute}) in feedforward model.")
    return hyu.instantiate(getattr(cfg, attribute), monads)\
        if cfg and attribute in cfg else torch.nn.Identity()

def _create_interval_block(
    cfg: omegaconf.DictConfig,
    force: bool=False
):
    if force and not cfg:
        log.warning("Empty interval block in feedforward model.")
    return hyu.instantiate(cfg) if cfg else NoInterval()

def _create_optimization_block(
    cfg: omegaconf.DictConfig,
    params: typing.Union[typing.Iterable[torch.Tensor], typing.Dict[str, torch.Tensor]],
    force: bool=False
):
    if force and not cfg:
        log.warning("No optimizer in feedforward model.")
    return hyu.instantiate(
        # cfg,
        #TODO: hack to get resolved values, should update when omegaconf 2.1 is out and explicit resolve is available
        omegaconf.OmegaConf.create(omegaconf.OmegaConf.to_container(cfg, resolve=True)),
        params
    ) if cfg else NoOptimization(params)

def _create_scheduling_block(
    cfg: omegaconf.DictConfig, 
    optimizers: typing.Sequence[torch.optim.Optimizer],
    force: bool=False
):
    if force and not cfg:
        log.warning("No scheduling used in feedforward model.")
    return hyu.instantiate(cfg, optimizers) if cfg else NoScheduling(optimizers)

def _assign_data(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    has_train = hasattr(cfg, 'train')
    has_test = hasattr(cfg, 'test')
    has_val = hasattr(cfg, 'val')
    if not has_train and not has_test and not has_val:
        log.error("No data have been included in the configuration. Please add the necessary \'- data/[split]/dataset/*: *\' entries.")
    if not has_test and has_val:
        test = omegaconf.OmegaConf.create({'test': cfg.val})
        omegaconf.OmegaConf.set_struct(cfg, False)
        cfg.merge_with(test)
        omegaconf.OmegaConf.set_struct(cfg, True)
        log.warning("No test dataset has been defined, using the validation dataset as a test dataset.")
    if not has_val and has_test:
        val = omegaconf.OmegaConf.create({'val': cfg.test})
        omegaconf.OmegaConf.set_struct(cfg, False)
        cfg.merge_with(val)
        omegaconf.OmegaConf.set_struct(cfg, True)
        log.warning("No validation dataset has been defined, using the test dataset as a validation dataset.")
    return cfg

class FeedForward(pytorch_lightning.LightningModule):
    def __init__(self, 
        data:               omegaconf.DictConfig=None,
        parameters:         omegaconf.DictConfig=None,
        feedforward:        omegaconf.DictConfig=None,
        monads:             omegaconf.DictConfig=None,        
        supervision:        omegaconf.DictConfig=None,
        validation:         omegaconf.DictConfig=None,        
        visualization:      omegaconf.DictConfig=None,
        export:             omegaconf.DictConfig=None,
        # hyperparameters:    typing.Union[omegaconf.DictConfig, typing.Mapping[str, typing.Any]]=None,
    ):
        super(FeedForward, self).__init__()
        self.data = _assign_data(data)
        self.initializer = parameters.initialization if parameters is not None else None
        # self.automatic_optimization = False
        self.optimization_config = parameters.optimization if parameters is not None else None
        self.schedule_config = parameters.schedule if parameters is not None else None
        self.schedule_monitor = parameters.schedule_monitor if parameters is not None and parameters.schedule_monitor is not None else None
        self.supervision = _create_supervision_block(supervision)
        self.validation = _create_validation_block(validation) #TODO: change this, "empty processing block" is confusing
        self.preprocess = _create_processing_block(feedforward, "preprocess", monads=monads)
        self.postprocess = _create_processing_block(feedforward, "postprocess", monads=monads)
        self.visualization = _create_interval_block(visualization)
        self.exporter = _create_interval_block(export)
        #NOTE: __NEEDED__ for loading checkpoint
        # hparams = hyperparameters if hyperparameters is not None else { }
        # hparams.update({'moai_version': miV})
        #NOTE: @PTL1.5 self.hparams =  hparams
        # self.hparams.update(hparams)
        self.global_test_step = 0
        # changes to work with PTL 2.0
        # self.validation_step_outputs = []
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        # self.training_step_outputs = [] TODO: check https://github.com/search?q=repo%3ALightning-AI%2Fpytorch-lightning%20on_train_batch_end%20&type=code

    def initialize_parameters(self) -> None:
        init = hyu.instantiate(self.initializer) if self.initializer else NoInit()
        init(self)

    def forward(self,
        tensors:                typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        pass

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        # PTL2.0 manual optimisation goes here
        # opt = self.optimizers()
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        postprocessed = self.postprocess(prediction)
        total_loss, losses = self.supervision(postprocessed)
        #TODO: should add loss maps as return type to be able to forward them for visualization
        losses = toolz.keymap(lambda k: f"train_{k}", losses)
        losses.update({'total_loss': total_loss})
        # opt.zero_grad()
        # self.manual_backward(total_loss)
        # opt.step()
        self.log_dict(losses, prog_bar=False, logger=True)
        # self.training_step_outputs.append(postprocessed)
        return { 'loss': total_loss, 'tensors': postprocessed }
    
    #TODO: should be removed for PTL 2.0
    # def training_step_end(self, 
    #     train_outputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]
    # ) -> None:
    #     if self.global_step and (self.global_step % self.visualization.interval == 0):
    #         self.visualization(train_outputs['tensors'], self.global_step)
    #     if self.global_step and (self.global_step % self.exporter.interval == 0):
    #         self.exporter(train_outputs['tensors'], self.global_step)
    #     return train_outputs['loss']

    def validation_step(self,
        batch:              typing.Dict[str, torch.Tensor],
        batch_nb:           int,
        dataloader_idx:   int=0,
    ) -> dict:        
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        outputs = self.postprocess(prediction)
        #TODO: consider adding loss maps in the tensor dict
        metrics = self.validation(outputs)
        # changes to work with PTL 2.0
        # append values to the dict with the same key as the dataloader
        self.validation_step_outputs[list(self.data.val.iterator.datasets.keys())[dataloader_idx]].append(metrics)
        # self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        # assume that ouputs is a dict
        # list_of_outputs = [outputs] if isinstance(toolz.get([0, 0], outputs)[0], dict) else outputs
        all_metrics = defaultdict(list)
        for i, dataset in enumerate(outputs):
            o = outputs[dataset]
            keys = next(iter(o), { }).keys()
            metrics = { }
            for key in keys:
                metrics[key] = np.mean(np.array(
                    [d[key].item() for d in o if key in d]
                ))
                all_metrics[key].append(metrics[key])
            log_metrics = toolz.keymap(lambda k: f"val_{k}/{dataset}", metrics)
            self.log_dict(log_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        all_metrics = toolz.valmap(lambda v: sum(v) / len(v), all_metrics)
        self.validation_step_outputs.clear()  # free memory
        self.log_dict(all_metrics, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
    
    def test_step(self, 
        batch: typing.Dict[str, torch.Tensor],
        batch_nb: int,
        dataloader_idx:   int=0, #NOTE check with None and kwargs
    ) -> dict:
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        outputs = self.postprocess(prediction)
        metrics = self.validation(outputs)
        self.test_step_outputs[list(self.data.test.iterator.datasets.keys())[dataloader_idx]].append(metrics)
        self.global_test_step += 1
        log_metrics = toolz.keymap(lambda k: f"test_{k}/{list(self.data.test.iterator.datasets.keys())[dataloader_idx]}", metrics)
        # check if iterator is zipped
        try:
            zp_target = self.data.test.iterator['_target_'].split(".")[-1]
        except:
            zp_target = None
        # TODO: update this part of code, and pass the dataloader index every time
        if self.data is not None and zp_target != 'Zipped' and len(self.data.test.iterator.datasets.keys()) > 1:
            #TODO: Check if PTL 2.0 does not support logging dictionaries, need to change this
            log_metrics.update({'dataloader_index': dataloader_idx})
            # log_metrics.update({'__moai__': {'dataloader_index': dataloader_idx}})
        self.log_dict(log_metrics, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return metrics, outputs

    # TODO: should be changed to on_test_batch_end in PTL 2.1
    def test_step_end(self,
        metrics_tensors: typing.Tuple[typing.Dict[str, torch.Tensor], typing.Dict[str, torch.Tensor]],
    ) -> None:
        metrics, tensors = metrics_tensors
        if self.global_test_step and (self.global_test_step % self.exporter.interval == 0):
            self.exporter(tensors, self.global_test_step)
        if self.global_test_step and (self.global_test_step % self.visualization.interval == 0):
            self.visualization(tensors, self.global_test_step)
        return metrics

    def on_test_epoch_end(self) -> dict:
        # caclulate the mean of the metrics per dataset
        #NOTE: do we need to calculate the mean of the metrics and saved them as a separate file?
        all_metrics = defaultdict(list)
        outputs = self.test_step_outputs
        log_metrics = defaultdict(list)
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
        #self.log_dict(log_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        for dataset in log_metrics.keys():
            ds = tablib.Dataset([v for v in log_metrics[dataset].values()], headers=log_metrics[dataset].keys())
            with open(os.path.join(os.getcwd(), f'{self.logger.name}_{dataset}_test_average.csv'), 'a', newline='') as f:
                 f.write(ds.export('csv'))
        self.test_step_outputs.clear()
    #TODO: test_epoch_end should be removed for PTL 2.0 to on_test_epoch_end
    # def test_epoch_end(self, 
    #     outputs: typing.List[dict]
    # ) -> dict:
    #     list_of_outputs = [outputs] if isinstance(toolz.get([0, 0], outputs)[0], dict) else outputs
    #     all_metrics = defaultdict(list)
    #     log_metrics = defaultdict(list)
    #     for i, o in enumerate(list_of_outputs):
    #         keys = next(iter(o), { }).keys()
    #         metrics = { }
    #         for key in keys:
    #             metrics[key] = np.mean(np.array(
    #                 [d[key].item() for d in o if key in d]
    #             ))
    #             all_metrics[key].append(metrics[key])
    #         log_metrics[list(self.data.test.iterator.datasets.keys())[i]] = metrics
    #     self.log_dict(log_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        log.info(f"Configuring optimizer and scheduler")
        self.optimization = _create_optimization_block(self.optimization_config, self.parameters())
        self.schedule = _create_scheduling_block(self.schedule_config, self.optimization.optimizers)
        if not self.schedule_monitor:
            return self.optimization.optimizers, self.schedule.schedulers
        else:
            return   {
                    'optimizer': self.optimization.optimizers[0],
                    'lr_scheduler': self.schedule.schedulers[0],
                    'monitor': self.schedule_monitor,
                }
        # return self.optimization.optimizers, self.schedule.schedulers if not self.schedule_monitor else \
        #        {
        #             'optimizer': self.optimization.optimizers,
        #             'lr_scheduler': self.schedule.schedulers,
        #             'monitor': self.schedule_monitor,
        #         }

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.train.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.train.iterator._target_.split('.')[-1]}) train set data iterator")
            train_iterator = hyu.instantiate(self.data.train.iterator)
        else:
            train_iterator = Indexed(
                self.data.train.iterator.datasets,
                self.data.train.iterator.augmentation if hasattr(self.data.train.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.train, 'loader'):
            log.error("Train data loader missing. Please add a data loader (i.e. \'- data/train/loader: torch\') entry in the configuration.")
        else:
            train_loader = hyu.instantiate(self.data.train.loader, train_iterator)
        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.val.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.val.iterator._target_.split('.')[-1]}) validation set data iterator")
            val_iterators = [hyu.instantiate(self.data.val.iterator)]
        else:
            val_iterators = [Indexed(
                {k: v }, # self.data.val.iterator.datasets,
                self.data.val.iterator.augmentation if hasattr(self.data.val.iterator, 'augmentation') else None,
            ) for k, v in self.data.val.iterator.datasets.items()]
        if not hasattr(self.data.val, 'loader'):
            log.error("Validation data loader missing. Please add a data loader (i.e. \'- data/val/loader: torch\') entry in the configuration.")
        else:
            validation_loaders = [
                hyu.instantiate(self.data.val.loader, val_iterator)
                for val_iterator in val_iterators
            ]
        # return validation_loaders[0] if len(validation_loaders) == 1 else validation_loaders
        return validation_loaders

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.test.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.test.iterator._target_.split('.')[-1]}) test set data iterator")
            test_iterators = [hyu.instantiate(self.data.test.iterator)]
            #test_iterator = hyu.instantiate(self.data.test.iterator)
        else:
            test_iterators = [Indexed(
                {k: v }, # self.data.val.iterator.datasets,
                self.data.test.iterator.augmentation if hasattr(self.data.test.iterator, 'augmentation') else None,
            ) for k, v in self.data.test.iterator.datasets.items()]
            # test_iterator = Indexed(
            #     self.data.test.iterator.datasets,
            #     self.data.test.iterator.augmentation if hasattr(self.data.test.iterator, 'augmentation') else None,
            # )
        if not hasattr(self.data.test, 'loader'):
            log.error("Test data loader missing. Please add a data loader (i.e. \'- data/test/loader: torch\') entry in the configuration.")
        else:
            test_loaders = [
                hyu.instantiate(self.data.test.loader, test_iterator)
                for test_iterator in test_iterators
            ]
            #test_loader = hyu.instantiate(self.data.test.loader, test_iterator)
        return test_loaders
