from moai import __version__ as miV

# from moai.utils.engine import NoOp as NoInterval
# from moai.parameters.optimization import NoOp as NoOptimization
# from moai.parameters.optimization.schedule import NoOp as NoScheduling
from moai.parameters.initialization import Default as NoInit
from moai.data.datasets.generic import Empty
# from moai.validation import (
#     NoOp as NoValidation,
#     Collection as DefaultValidation,
# )
# from moai.supervision import (
#     NoOp as NoSupervision,
#     Weighted as DefaultSupervision,
# )
from moai.data.iterator import Indexed
# from moai.monads.execution.cascade import _create_accessor
# from moai.networks.lightning._optimizer import (
#     _create_assigner,
#     _create_supervision_block,
#     _create_validation_block,
# )
# from moai.validation.noop import NoOp as NoValidation
from moai.validation.collection import Metrics as DefaultValidation

# from moai.supervision.noop import NoOp as NoSupervision
from moai.supervision.weighted import Weighted as DefaultSupervision
from moai.utils.iterators import partition
from moai.utils.funcs import (
    select, select_dict, select_list,
    get, get_dict, get_list,
)
from moai.utils.arguments import ensure_string_list

from moai.core.execution.monads import Monads
from moai.core.execution.tensors import Tensors
from moai.core.execution.criteria import Criteria
from moai.core.execution.models import Models
from moai.core.execution.constants import Constants

from collections import defaultdict, OrderedDict

from pytorch_lightning.trainer import call
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior

import moai.core.execution.common as mic
import torch
import pytorch_lightning as L
import omegaconf.omegaconf
import hydra.utils as hyu
import typing
import toolz
import logging
import functools
import benedict

log = logging.getLogger(__name__)

__all__ = ['Model']

#TODO: vary w.r.t mode infer/predict to change clone/copy behaviour
def _create_assigner(key: str) -> typing.Callable[[torch.nn.Module, torch.Tensor], None]:
    key_list = [key] if isinstance(key, str) else key
    splits = [k.split('.') for k in key_list]
    def _assign(m: torch.nn.Module, t: torch.Tensor, keys: typing.Sequence[str]):
        for split in keys:
            to_set = toolz.reduce(getattr, split, m)
            to_set.copy_(t.clone())
            # to_set.copy_(t)    
    return functools.partial(_assign, keys=splits)

class MoaiLightningModule(L.LightningModule):
    def __init__(self,
        modules:            omegaconf.DictConfig=None,
        monads:             omegaconf.DictConfig=None,
        parameters:         omegaconf.DictConfig=None,
        # graphs:             omegaconf.DictConfig=None,#TODO: remove        
        objectives:         omegaconf.DictConfig=None,
        metrics:            omegaconf.DictConfig=None,
        monitors:           omegaconf.DictConfig=None,
        modifications:      omegaconf.DictConfig=None,        
        # monitoring:         omegaconf.DictConfig=None,#TODO: remove
        # stopping:           omegaconf.DictConfig=None,
        hyperparameters:    typing.Union[omegaconf.DictConfig, typing.Mapping[str, typing.Any]]=None,
        data:               omegaconf.DictConfig=None,
        _moai_:             omegaconf.DictConfig=None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.data = data
        ## Check for generation metrics
        #TODO: how to integrate?
        '''
        self.generation_metrics = {}
        if toolz.get_in(['generation'], validation) is not None:
            gen_metrics = validation['generation']['metrics'].keys()
            groups = list(monitors.metrics.keys())
            for m in list(gen_metrics):
                for g in groups:
                    tmp = toolz.get_in([m], monitors.metrics[g]) or {}
                    if tmp:
                        self.generation_metrics[m] = tmp
        '''
        ## Inner modules aka Models
        self.models = torch.nn.ModuleDict()
        for k in modules or {}:
            self.models[k] = hyu.instantiate(modules[k])
        ## Monad & Module Processing Graphs
        self.named_flows = torch.nn.ModuleDict()
        flows = select_dict(_moai_, Constants._FLOWS_)
        monad_flows, model_flows = partition(lambda k: k in self.models, flows or {})
        for model_flow in model_flows:
            self.named_flows[model_flow] = Models(models=self.models, **{model_flow: flows[model_flow]})
        for monad_flow in monad_flows:
            self.named_flows[monad_flow] = Monads(monads=monads, **flows[monad_flow])
        ## Objectives
        self.named_objectives = torch.nn.ModuleDict()
        for k, v in select_dict(_moai_, Constants._OBJECTIVES_COLLECTION_).items():
        # for k in omegaconf.OmegaConf.select(parameters,"optimization.objectives") or {}:
            # v = parameters.optimization.objectives[k]
            self.named_objectives[k] = DefaultSupervision(#_create_supervision_block(
                objectives, **v
                # omegaconf.OmegaConf.merge(
                #     omegaconf.OmegaConf.masked_copy(_moai_, Constants._OBJECTIVES_), 
                #     v
                # )
            )
        ## Metrics Monitors
        self.named_metrics = torch.nn.ModuleDict()
        self.metric_name_to_module = torch.nn.ModuleDict()
        # for k in toolz.get_in(['metrics'], monitors) or {}:
        for k, v in select_dict(_moai_, Constants._METRICS_COLLECTION_).items():
            # v = monitors.metrics[k]
            self.named_metrics[k] = DefaultValidation( #_create_validation_block(
                metrics, **v
                # omegaconf.OmegaConf.merge(validation, v)
            )
            for key, out, _ in self.named_metrics[k].execs:
                if out in self.metric_name_to_module:
                    log.error(f"Same metric name [{out}] used in multiple definitions, metrics will not compute correctly!")
                self.metric_name_to_module[out] = self.named_metrics[k][key]
        #NOTE: use masked_copy to keep only metrics w/ the key
        ## Tensor Monitors
        self.named_monitors = {}
        # for k in toolz.get_in(['tensors'], monitors) or {}:
        for k, v in select_dict(_moai_, Constants._MONITORS_COLLECTION_).items():
            # v = monitors.tensors[k]
            self.named_monitors[k] = Tensors(#_create_tensor_monitoring_block(
                monitors, **v
                # omegaconf.OmegaConf.merge(v, monitoring) #NOTE: for some reason the order needs to be reverted
            )
        ## Termination Criteria
        self.named_criteria = {}
        # for k in toolz.get_in(['criteria'], monitors) or {}:
        for k, v in select_dict(_moai_, Constants._CRITERIA_COLLECTION_).items():
            # v = monitors.criteria[k]
            self.named_criteria[k] = Criteria(#_create_criteria_monitoring_block(
                select(parameters, Constants._CRITERIA_), **v
                # omegaconf.OmegaConf.merge(v, stopping) #NOTE: for some reason the order needs to be reverted
            )
        #NOTE: up to here anything with optimizable parameters that can be selected
        #       should be created and available in `self`
        ## Optimizers & Parameters
        self.named_optimizers = OrderedDict()
        # for k in toolz.get_in(["optimization", "optimizers", "named"], parameters) or {}:
        for k, v in select_dict(_moai_, Constants._OPTIMIZERS_COLLECTION_).items():
            # v = parameters.optimization.optimizers.named[k]
            # if isinstance(v.selectors, str):
            #     selectors = [parameters.selectors[v.selectors]]
            # else:
            #     selectors = [parameters.selectors[s] for s in v.selectors]
            groups = [select(parameters, Constants._GROUPS_)[g] for g in ensure_string_list(select(v, 'groups'))]
            # optimizer = parameters.optimization.optimizers[select(v, 'type')]
            optimizer = select(parameters, Constants._OPTIMIZERS_)[select(v, 'type')]
            config = omegaconf.OmegaConf.merge(optimizer, select_dict(v, 'params'))
            selected_params = list(hyu.instantiate(g)(self) for g in groups)
            self.named_optimizers[k] = hyu.instantiate(config, selected_params)
        #TODO: mode should be selected from the config and call the 
        # correct initalizer
        self.named_initializers = defaultdict(list)
        # for k, v in select_dict(_moai_, Constants._INITIALIZERS_COLLECTION_).items():
        for k, v in omegaconf.OmegaConf.to_container(
            select(_moai_, Constants._INITIALIZE_)[select(_moai_, "_action_")],#parameters.optimization.process, 
            resolve=True
        ).items():
            # get initializers for each group (e.g. setup, batch, epoch)
            self.named_initializers[k] = list(hyu.instantiate(parameters.initializers[v])) if isinstance(v, str) \
                else list(hyu.instantiate(parameters.initializers[i]) for i in v)

        ## Schedulers
        self.named_schedulers = defaultdict(dict) # {'step': {}, 'epoch': {}}
        # for k in toolz.get_in(["optimization", "schedulers", "named"], parameters) or {}:
        for k, v in select_dict(_moai_, Constants._SCHEDULERS_COLLECTION_).items():
            # v = parameters.optimization.schedulers.named[k]
            scheduler = parameters.optimization.schedulers[select(v, 'type')]
            config = omegaconf.OmegaConf.merge(scheduler, select_dict(v, 'params'))
            interval = select(v, 'interval') or 'epoch' #NOTE: if missing defaults to `epoch`
            self.named_schedulers[interval][k] = hyu.instantiate(config, self.named_optimizers[select(v, 'optimizer')])
        ## Optimization Process & Monitoring
        self.process = omegaconf.OmegaConf.to_container(
            select(_moai_, Constants._EXECUTION_),#parameters.optimization.process, 
            resolve=True
        )
        self.monitor = omegaconf.OmegaConf.to_container(
            select(_moai_, Constants._MONITORING_), # parameters.optimization.monitor, 
            resolve=True
        )
        # Aggregate results
        self.test_step_outputs = defaultdict(list)
        self.scalar_metrics = defaultdict(list)
        self.non_scalar_metrics = defaultdict(list)
        #NOTE: __NEEDED__ for loading checkpoint?
        hparams = hyperparameters if hyperparameters is not None else { }
        hparams.update({'moai_version': miV})
        self.hparams.update(hparams)
        # remodel
        for name, remodel in (modifications or {}).items():
            log.info(f"Modifying the model with {name}.")
            hyu.instantiate(remodel)(self)
        #TODO: init
        
    def setup_initializers(self) -> None:
        # call the initializers once at the beginning
        # get the initializers to be called at the beginning
        for init in self.named_initializers['setup']:
            init(self)
    
    def batch_initializers(self) -> None:
        # call the initializers at the beginning of each batch
        for init in self.named_initializers['batch']:
            init(self)
    
    def epoch_initializers(self) -> None:
        # call the initializers at the beginning of each epoch
        for init in self.named_initializers['epoch']:
            init(self)

    #TODO: deprecated? to be removed
    def initialize_parameters(self) -> None:
        init = hyu.instantiate(self.initializer) if self.initializer else NoInit()
        init(self)
    
    def _assign_params(self,
        assigners: typing.List[typing.Tuple[str, functools.partial]],
        tensors: typing.Dict[str, torch.Tensor]
    ) -> None:
        for i, a in assigners:
            accessor = mic._create_accessor(i)
            a(self, accessor(tensors))

    def predict_step(self,
        batch:  typing.Dict[str, torch.Tensor],
        batch_idx: int,
        dataset_idx: int=0,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        log.info(f"Predicting batch {batch_idx} ...")
        batch = benedict.benedict(batch, keyattr_enabled=False)
        monitor = toolz.get_in(['predict', 'batch'], self.monitor) or []
        for stage, proc in self.process['predict']['batch'].items():
            steps = proc['steps']
            with torch.no_grad(): #TODO: probably this is not needed
                # for iter in range(iters): #NOTE: is this necessary?
                for step in steps:
                    batch = self.named_flows[step](batch)
                # predict step does 
                if monitor:
                    # Metrics monitoring used only for serve
                    for metric in toolz.get('metrics', monitor, None) or []:
                        self.named_metrics[metric](batch)
                    # Tensor monitoring for visualization & exporting
                    tensor_monitors = toolz.get('tensors', monitor, None) or []
                    for tensor_monitor in tensor_monitors:
                        extras = {
                            'stage': 'predict',
                            'step': self.global_step,
                            'batch_idx': batch_idx,
                            'optimization_step': 0, #TODO: add this for fitting case
                        }
                        #TODO: What extras should be passed here?
                        self.named_monitors[tensor_monitor](batch, extras)

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:                
        def closure(tensors, index, steps, stage, optimizer, objective):
            # def backward_fn(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
                # call._call_strategy_hook(self.trainer, "backward", loss, optimizer)        
            for step in steps:
                tensors = self.named_flows[step](tensors)
            loss, losses = self.named_objectives[objective](tensors)
            is_first_batch_to_accumulate = index % self.trainer.accumulate_grad_batches == 0
            if self.trainer.accumulate_grad_batches == 1 or not is_first_batch_to_accumulate:
                call._call_callback_hooks(self.trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(self.trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(self.trainer, "optimizer_zero_grad", self.trainer.current_epoch, index, optimizer)
            call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
            self.optimization_step += 1
            # for k in toolz.get_in(['fit', 'step', stage], self.monitor) or {}:
                # monitor = self.monitor['fit']['step'][k]
            monitor = toolz.get_in(['fit', 'step', stage], self.monitor)
            if monitor is not None:
                tensor_monitor_steps = toolz.get_in(['tensors'], monitor) or []
                if tensor_monitor_steps and self.optimization_step % monitor['frequency'] == 0:
                    with torch.no_grad():
                        for step in toolz.get_in(['steps'], monitor) or []:
                            self.named_flows[step](tensors)
                        extras = {
                            'step': self.global_step, 'epoch': self.current_epoch,
                            'optimization_step': self.optimization_step,
                            'batch_idx': batch_idx, 'stage': stage,
                        }
                        for step in tensor_monitor_steps:
                            self.named_monitors[step](tensors, extras)
            return loss
        batch = benedict.benedict(batch, keyattr_enabled=False)
        batch[Constants._MOAI_METRICS_] = {}
        batch[Constants._MOAI_LOSSES_] = {}
        #TODO: check for refresh optimizers each step
        for stage, proc in self.process['fit']['batch'].items():
            steps = proc['steps']
            iters = proc.get('iterations', 1)
            optim = proc.get('optimizer', None)
            assign_params = proc.get('assign', None)
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
                                batch = self.named_flows[step](batch)
                iter_monitor_stage = toolz.get_in(['fit', 'iter', stage], self.monitor)
                if iter_monitor_stage is not None:
                    # for _, iter_monitor_stage in iter_monitor.items():                        
                    frequency = toolz.get('frequency', iter_monitor_stage, 1)
                    should_monitor = iter % frequency == 0
                    iter_tensor_monitor = toolz.get('tensors', iter_monitor_stage)
                    if should_monitor and iter_tensor_monitor is not None:
                        for step in toolz.get('steps', iter_monitor_stage, None) or []:
                            self.named_flows[step](batch)
                        for metric in toolz.get('metrics', iter_monitor_stage, None) or []:
                            self.named_metrics[metric](batch)
                        extras = {
                            'step': self.global_step, 'epoch': self.current_epoch,
                            'batch_idx': batch_idx, 'stage': stage, 'iter': iter
                        }
                        for step in iter_tensor_monitor:
                            self.named_monitors[step](batch, extras)
                        should_stop = False
                        for criterion in toolz.get('termination', iter_monitor_stage, None) or []:
                            if self.named_criteria[criterion](batch, extras):
                                should_stop = True
                                break
                        if should_stop:
                            log.info(f"Terminating {stage} @ {iter} !")
                            break
            # call the copy params for initialization
            if assign_params is not None:
                frequency = assign_params.get('frequency', 1) # default to each batch end
                assigners = []
                if batch_idx == 0: # get initializers only in the first batch                    
                    for i, o in assign_params.items():
                        if i == 'frequency': #NOTE: refactor this, keys should be not coupled like this
                            continue
                        assigners.append((i, _create_assigner(o)))                
                with torch.no_grad(): # use torch no grad as most params are leaf tensors and assign is an inplace operation
                    if frequency == 0: # if frequency is 0 call only once
                        if batch_idx == 0:
                            self._assign_params(assigners, batch)
                    else:
                        if batch_idx % frequency == 0:
                            self._assign_params(assigners, batch)
        return batch

    @torch.no_grad()
    def test_step(self,
        batch:              typing.Dict[str, torch.Tensor],
        batch_nb:           int,
        dataloader_idx:     int=0,
    ) -> dict:
        batch = benedict.benedict(batch, keyattr_enabled=False)
        batch[Constants._MOAI_METRICS_] = {}
        datasets = list(self.data.test.iterator.datasets.keys())
        monitor = toolz.get_in(['test', 'batch'], self.monitor) or []
        # get graphs for test
        for stage, proc in self.process['test']['batch'].items():
            steps = proc['steps']
            with torch.no_grad(): #TODO: probably this is not needed
                # for iter in range(iters): #NOTE: is this necessary?
                for step in steps:
                    batch = self.named_flows[step](batch)
                if monitor:
                    # Metrics monitoring
                    for metric in toolz.get('metrics', monitor, None) or []:
                        self.named_metrics[metric](batch)
                    # Tensor monitoring for visualization
                    tensor_monitors = toolz.get('tensors', monitor, None) or []
                    for tensor_monitor in tensor_monitors:
                        self.named_monitors[tensor_monitor](batch)

    @torch.no_grad
    def validation_step(self,
        batch:              typing.Dict[str, torch.Tensor],
        batch_nb:           int,
        dataloader_idx:   int=0,
    ) -> None:
        batch = benedict.benedict(batch, keyattr_enabled=False)
        batch[Constants._MOAI_METRICS_] = {}
        if not hasattr(self.data, 'val'):
            log.warning("Validation data missing. An empty validation set will be used.")
            return
        datasets = list(self.data.val.iterator.datasets.keys())
        monitor = toolz.get_in(['val', 'batch'], self.monitor)
        for stage, proc in (toolz.get_in(['val', 'batch', datasets[dataloader_idx]], self.process, {}) or {}).items():
            steps = proc['steps']
            with torch.no_grad():
                for step in steps:
                    batch = self.named_flows[step](batch)
                for _, monitor_stage in monitor.items():
                    for metric in toolz.get('metrics', monitor_stage, None) or []:
                        self.named_metrics[metric](batch) #TODO add visualization
        return batch

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        return list(self.named_optimizers.values()), list(self.named_schedulers.values())

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.train.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.train.iterator._target_.split('.')[-1]}) train set data iterator")
            train_iterator = hyu.instantiate(self.data.train.iterator, _recursive_=False)
        else:
            train_iterator = Indexed(
                self.data.train.iterator.datasets,
                self.data.train.iterator.augmentation if hasattr(self.data.train.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.train, 'loader'):
            log.error("Train data loader missing. Please add a data loader (i.e. \'- data/train/loader: torch\') entry in the configuration.")
        else:
            train_loader = hyu.instantiate(self.data.train.loader, train_iterator, _recursive_=False)
        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # check if key val is in struct
        if not hasattr(self.data, 'val'):
            log.warning("Validation data missing. An empty validation set will be used.")
            validation_loaders = [torch.utils.data.DataLoader(Empty())]
            return validation_loaders
        if hasattr(self.data.val.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.val.iterator._target_.split('.')[-1]}) validation set data iterator")
            val_iterators = [hyu.instantiate(self.data.val.iterator, _recursive_=False)]
        else:
            val_iterators = [Indexed(
                {k: v }, # self.data.val.iterator.datasets,
                self.data.val.iterator.augmentation if hasattr(self.data.val.iterator, 'augmentation') else None,
            ) for k, v in self.data.val.iterator.datasets.items()]
        if not hasattr(self.data.val, 'loader'):
            log.error("Validation data loader missing. Please add a data loader (i.e. \'- data/val/loader: torch\') entry in the configuration.")
        else:
            validation_loaders = [
                hyu.instantiate(self.data.val.loader, val_iterator, _recursive_=False)
                for val_iterator in val_iterators
            ]
        # return validation_loaders[0] if len(validation_loaders) == 1 else validation_loaders
        return validation_loaders

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        #NOTE: instead of predict loader use testing or val?
        # check if key val is in struct
        if not hasattr(self.data, 'predict'):
            log.warning("Predict data missing. An empty predict set will be used.")
            predict_loaders = [torch.utils.data.DataLoader(Empty())]
            return predict_loaders
        if hasattr(self.data.predict.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.predict.iterator._target_.split('.')[-1]}) prediction set data iterator")
            predict_iterators = [hyu.instantiate(self.data.predict.iterator, _recursive_=False)]
        else:
            predict_iterators = [Indexed(
                {k: v },
                self.data.predict.iterator.augmentation if hasattr(self.data.predict.iterator, 'augmentation') else None,
            ) for k, v in self.data.predict.iterator.datasets.items()]
        if not hasattr(self.data.predict, 'loader'):
            log.error("Prediction data loader missing. Please add a data loader (i.e. \'- data/predict/loader: torch\') entry in the configuration.")
        else:
            prediction_loaders = [
                hyu.instantiate(self.data.predict.loader, predict_iterator, _recursive_=False)
                for predict_iterator in predict_iterators
            ]
        return prediction_loaders

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