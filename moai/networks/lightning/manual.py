#NOTE: hparams vs hyper_parameters

import pytorch_lightning.trainer
from moai import __version__ as miV
from moai.utils.engine import NoOp as NoInterval
from moai.parameters.optimization import NoOp as NoOptimization
from moai.parameters.optimization.schedule import NoOp as NoScheduling
from moai.parameters.initialization import Default as NoInit
from moai.data.datasets.generic import Empty
from moai.validation import (
    NoOp as NoValidation,
    Collection as DefaultValidation,
)
from moai.supervision import (
    NoOp as NoSupervision,
    Weighted as DefaultSupervision,
)
from moai.data.iterator import Indexed
from moai.parameters.optimization.optimizers import LBFGS as miLBFGS
from moai.monads.execution.cascade import _create_accessor
from moai.networks.lightning.optimizer import (
    _create_optimization_block,
    _create_interval_block,
    _create_assigner,
    _create_processing_block,
    _create_scheduling_block,
    _create_supervision_block,
    _create_validation_block,
)
import moai.utils.parsing.rtp as mirtp
from moai.utils.iterators import partition

from moai.monads.execution.cascade import Cascade
from moai.monads.execution.models import Models, Tensors

from collections import defaultdict, OrderedDict

from pytorch_lightning.trainer import call
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior

import torch
import pytorch_lightning
import omegaconf.omegaconf
import hydra.utils as hyu
import typing
import toolz
import logging
import inspect
import functools

log = logging.getLogger(__name__)

__all__ = ['Manual']

def _create_tensor_monitoring_block(
    cfg: omegaconf.DictConfig,
    force: bool=True
):
    if force and not cfg:
        log.warning("Empty tensor monitoring block in feedforward model.")    
    # if not cfg:
    #     return NoValidation()
    if '_target_' not in cfg:
        return Tensors(**cfg)
    else:
        return hyu.instantiate(cfg)

class Manual(pytorch_lightning.LightningModule):
    def __init__(self,
        configuration:      omegaconf.DictConfig,
        modules:            omegaconf.DictConfig=None,
        data:               omegaconf.DictConfig=None,
        parameters:         omegaconf.DictConfig=None,
        graphs:             omegaconf.DictConfig=None,
        monads:             omegaconf.DictConfig=None,
        supervision:        omegaconf.DictConfig=None,
        validation:         omegaconf.DictConfig=None,
        monitors:           omegaconf.DictConfig=None,
        monitoring:         omegaconf.DictConfig=None,
        # visualization:      omegaconf.DictConfig=None,
        # export:             omegaconf.DictConfig=None,
        hyperparameters:    typing.Union[omegaconf.DictConfig, typing.Mapping[str, typing.Any]]=None,
    ):
        super().__init__()
        self.automatic_optimization = False
        # create moduledict of named models
        # create moduledict of model specific pre/post monad proc graph
        # create schedulers & optimizers
        # assemble optimization process
        # use scopes for models/monads
        self.data = data
        self.models = torch.nn.ModuleDict()
        for k in modules or {}:            
            self.models[k] = hyu.instantiate(modules[k])
        self.graphs = torch.nn.ModuleDict()
        monad_graphs, model_graphs = partition(lambda k: k in self.models, graphs or {})
        for model_graph in model_graphs:
            self.graphs[model_graph] = Models(models=self.models, **{model_graph: graphs[model_graph]})
        for monad_graph in monad_graphs:
            self.graphs[monad_graph] = Cascade(monads=monads, **graphs[monad_graph])

        # for k in graphs or {}:
        #     if k in self.models:
        #         self.graphs[k] = Models(models=models, **graphs[k])
        #     else:
        #         self.graphs[k] = Cascade(monads=monads, **graphs[k])  
        self.named_objectives = torch.nn.ModuleDict()
        for k in parameters.optimization.objectives or {}:
            v = parameters.optimization.objectives[k]
            self.named_objectives[k] = _create_supervision_block(
                omegaconf.OmegaConf.merge(supervision, v)
            )
        self.named_metrics = torch.nn.ModuleDict()
        for k in monitors.metrics or {}:
            v = monitors.metrics[k]
            self.named_metrics[k] = _create_validation_block(
                omegaconf.OmegaConf.merge(validation, v)
            )
        self.named_monitors = {}
        for k in monitors.tensors or {}:
            v = monitors.tensors[k]
            self.named_monitors[k] = _create_tensor_monitoring_block(
                omegaconf.OmegaConf.merge(v, monitoring) #NOTE: for some reason the order needs to be reverted
            )
        #NOTE: up to here anything with optimizable parameters that can be selected
        #       should be created and available in `self`
        self.named_optimizers = OrderedDict()
        for k in toolz.get_in(["optimization", "optimizers", "named"], parameters) or {}:
            v = parameters.optimization.optimizers.named[k]
            if isinstance(v.selectors, str):
                selectors = [parameters.selectors[v.selectors]]
            else:
                selectors = [parameters.selectors[s] for s in v.selectors]
            optimizer = parameters.optimization.optimizers[v.type]
            config = omegaconf.OmegaConf.merge(optimizer, v.params)
            selected_params = list(hyu.instantiate(s)(self) for s in selectors)
            self.named_optimizers[k] = hyu.instantiate(config, selected_params)
        self.named_schedulers = defaultdict(dict) # {'step': {}, 'epoch': {}}
        for k in toolz.get_in(["optimization", "schedulers", "named"], parameters) or {}:
            v = parameters.optimization.schedulers.named[k]
            scheduler = parameters.optimization.schedulers[v.type]
            config = omegaconf.OmegaConf.merge(scheduler, v.params)
            interval = v.interval or 'epoch'
            self.named_schedulers[interval][k] = hyu.instantiate(config, self.optimizers[v.optimizer])
        self.process = omegaconf.OmegaConf.to_container(parameters.optimization.process, resolve=True)
        self.monitor = omegaconf.OmegaConf.to_container(parameters.optimization.monitor, resolve=True)
        self.initializer = parameters.initialization if parameters is not None else None
        #NOTE: __NEEDED__ for loading checkpoint?
        hparams = hyperparameters if hyperparameters is not None else { }
        hparams.update({'moai_version': miV})
        #NOTE: @PTL1.5 self.hparams =  hparams
        self.hparams.update(hparams)
        
    def initialize_parameters(self) -> None:
        init = hyu.instantiate(self.initializer) if self.initializer else NoInit()
        init(self)
    
    def copy_params(self,
        initializers: typing.List[typing.Tuple[str, functools.partial]],
        tensors: typing.Dict[str, torch.Tensor]
    ) -> None:
        for i, a in initializers:
            accessor = _create_accessor(i)
            a(self,accessor(tensors))

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        # pass
        # closure = self._make_closure(kwargs, optimizer, batch_idx)
        # call._call_strategy_hook(self.trainer, "backward", loss, optimizer)

        # training_step -> from process current stage steps & supervision
        # post train step (happens in manual optimization) => BUT accumulate grad batch normalization DOES NOT
        # use automatic optimization and override _training_step
        # then add step monitor (stop criteria) & step schedulers
        # 1. make closure
        #       a. custom inner step (based on proc steps & loss)
        #       b. backward & zero grad from automatic optimization? => these call hooks
        #               backward_fn should accept both loss & optimizer
        # 2. check accumulation from trainer's fit loop
        # 3. call trainer's optimizer_step /w current optimizer and closure
        #           !!! it is important for the tensor dict to be updated inplace
        #           !!! to make sure the intermediate results are available to the following ops
        # 4. the batch gets deleted from memory after the training step
        # 5. should return a dict /w `loss`
        def closure(tensors, index, steps, stage, optimizer, objective):
            # step_fn = self._make_step_fn(kwargs)            
            # def backward_fn(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
                # call._call_strategy_hook(self.trainer, "backward", loss, optimizer)        
            for step in steps:
                tensors = self.graphs[step](tensors)
            loss, losses = self.named_objectives[objective](tensors)
            is_first_batch_to_accumulate = index % self.trainer.accumulate_grad_batches == 0
            if self.trainer.accumulate_grad_batches == 1 or not is_first_batch_to_accumulate:
                call._call_callback_hooks(self.trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(self.trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(self.trainer, "optimizer_zero_grad", self.trainer.current_epoch, index, optimizer)
            call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
            self.optimization_step += 1
            for k in toolz.get_in(['fit', 'step'], self.monitor) or {}:
                monitor = self.monitor['fit']['step'][k]
                tensor_monitor_steps = toolz.get_in(['tensors'], monitor) or []                
                if tensor_monitor_steps and self.optimization_step % monitor['frequency'] == 0:
                    with torch.no_grad():
                        for step in toolz.get_in(['steps'], monitor) or []:
                            self.graphs[step](tensors)
                        extras = {
                            'step': self.global_step, 'epoch': self.current_epoch,
                            'optimization_step': self.optimization_step,
                            'batch_idx': batch_idx, 'stage': stage,
                        }
                        for step in tensor_monitor_steps:
                            self.named_monitors[step](tensors, extras)
            return loss
        #TODO: check for refresh optimizers each step
        # self.log_dict()

        for stage, proc in self.process['fit']['batch'].items():
            steps = proc['steps']
            iters = proc.get('iterations', 1)
            optim = proc.get('optimizer', None)
            copy_params = proc.get('copy_params', None)
            if optim is not None:
                optimizers = self.optimizers()
                if isinstance(optimizers, list):
                    optimizer = optimizers[list(self.named_optimizers.keys()).index(optim)]
                else:
                    if list(self.named_optimizers.keys()).index(optim) == 0:
                        optimizer = optimizers
                    else:
                        log.warning(f"Optimizer {optim} with index {list(self.named_optimizers.keys()).index(optim)} is not found!")
            else:
                optimizer = None
            # optimizer = self.optimizers()[list(self.named_optimizers.keys()).index(optim)]\
                # if optim is not None else None #NOTE: this can be cached at ctor
            objective = proc.get('objective', None)
            current_closure = functools.partial(closure, batch, batch_idx, steps, stage, optimizer, objective)
            for iter in range(iters):
                log.info(f"Training step {iter+1}/{iters} for {k} with {steps} steps")
                if (# when the strategy handles accumulation, we want to always call the optimizer step
                    not self.trainer.strategy.handles_gradient_accumulation and self.trainer.fit_loop._should_accumulate()
                ): # For gradient accumulation calculate loss (train step + train step end)
                    with _block_parallel_sync_behavior(self.trainer.strategy, block=True):
                        current_closure() # automatic_optimization=True: perform ddp sync only when performing optimizer_step
                else:
                    if optimizer is not None:
                        self.optimizer_step(self.trainer.current_epoch, batch_idx, optimizer, current_closure)
                    else: #NOTE: w/o an optim, it is a tensor setup step (e.g. inference)
                        with torch.no_grad():
                            # for iter in range(iters): #NOTE: is this necessary?
                            for step in steps:
                                batch = self.graphs[step](batch)
                iter_monitor = toolz.get_in(['fit', 'iter'], self.monitor)
                if iter_monitor is not None:
                    for _, iter_monitor_stage in iter_monitor.items():                        
                        frequency = toolz.get('frequency', iter_monitor_stage, 1)
                        should_monitor = iter % frequency == 0
                        iter_tensor_monitor = toolz.get('tensors', iter_monitor_stage)
                        if should_monitor and iter_tensor_monitor is not None:
                            for step in toolz.get('steps', iter_monitor_stage, None) or []:
                                self.graphs[step](batch)
                            for metric in toolz.get('metrics', iter_monitor_stage, None) or []:
                                self.named_metrics[metric](batch)
                            extras = {
                                'step': self.global_step, 'epoch': self.current_epoch,
                                'batch_idx': batch_idx, 'stage': stage, 'iter': iter
                            }
                            for step in iter_tensor_monitor:
                                self.named_monitors[step](batch, extras)

            # call the copy params for initialization
            if copy_params is not None:
                frequency = copy_params.get('frequency', 1) # default to each batch end
                initializers = []
                if batch_idx == 0:
                    # get initializers only in the first batch
                    for i, o in copy_params.items():
                        if i == 'frequency':
                            continue
                        initializers.append((i, _create_assigner(o)))
                # if frequency is 0 call only once
                # use torch no grad as most params
                # are leaf tensors and assign is an inplace operation
                with torch.no_grad():
                    if frequency == 0:
                        if batch_idx == 0:
                            self.copy_params(initializers, batch)
                    else:
                        if batch_idx % frequency == 0:
                            self.copy_params(initializers, batch)
        return batch
            

    def validation_step(self,
        batch:              typing.Dict[str, torch.Tensor],
        batch_nb:           int,
        dataloader_idx:   int=0,
    ) -> dict:        
        pass

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        return list(self.named_optimizers.values()), list(self.named_schedulers.values())

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
        # check if key val is in struct
        if not hasattr(self.data, 'val'):
            log.warning("Validation data missing. An empty validation set will be used.")
            validation_loaders = [torch.utils.data.DataLoader(Empty())]
            return validation_loaders
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