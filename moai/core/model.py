from moai import __version__ as miV

from moai.parameters.initialization import Default as NoInit
from moai.data.datasets.generic import Empty
from moai.data.iterator import Indexed
from moai.validation.collection import Metrics as DefaultValidation
from moai.supervision.weighted import Weighted as DefaultSupervision
from moai.utils.iterators import partition
from moai.utils.funcs import (
    select, select_dict, select_list, select_conf,
    get, get_dict, get_list,
)
from moai.utils.arguments import ensure_string_list
from moai.core.execution.monads import Monads
from moai.core.execution.tensors import Tensors
from moai.core.execution.criteria import Criteria
from moai.core.execution.models import Models
from moai.core.execution.constants import Constants as C

from collections import defaultdict, OrderedDict, deque

from pytorch_lightning.trainer import call
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior

from omegaconf.omegaconf import DictConfig, OmegaConf

import moai.core.execution.common as mic
import torch
import pytorch_lightning as L
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
        modules:            DictConfig=None,
        monads:             DictConfig=None,
        parameters:         DictConfig=None,       
        objectives:         DictConfig=None,
        metrics:            DictConfig=None,
        monitors:           DictConfig=None,
        modifications:      DictConfig=None,        
        hyperparameters:    typing.Union[DictConfig, typing.Mapping[str, typing.Any]]=None,
        data:               DictConfig=None,
        _moai_:             DictConfig=None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.data = data        
        ## Inner modules aka Models
        self.models = torch.nn.ModuleDict()
        for k in modules or {}:
            self.models[k] = hyu.instantiate(modules[k])
        ## Monad & Module Processing Graphs
        self.named_flows = torch.nn.ModuleDict()
        flows = select_dict(_moai_, C._DEFINED_FLOWS_)
        monad_flows, model_flows = partition(lambda k: k in self.models, flows or {})
        for model_flow in model_flows:
            self.named_flows[model_flow] = Models(models=self.models, **{model_flow: flows[model_flow]})
        for monad_flow in monad_flows:
            self.named_flows[monad_flow] = Monads(monads=monads, **flows[monad_flow])
        ## Objectives
        self.named_objectives = torch.nn.ModuleDict()
        for k, v in select_dict(_moai_, C._OBJECTIVES_COLLECTION_).items():
            self.named_objectives[k] = DefaultSupervision(
                objectives, **v
            )
        ## Metrics Monitors
        self.named_metrics = torch.nn.ModuleDict()
        self.metric_name_to_module = torch.nn.ModuleDict()
        for k, v in select_dict(_moai_, C._METRICS_COLLECTION_).items():
            self.named_metrics[k] = DefaultValidation(
                metrics, **v
            )
            for key, out, _ in self.named_metrics[k].execs:
                if out in self.metric_name_to_module:
                    log.error(f"Same metric name [{out}] used in multiple definitions, metrics will not compute correctly!")
                self.metric_name_to_module[out] = self.named_metrics[k][key]
        #NOTE: use masked_copy to keep only metrics w/ the key
        ## Tensor Monitors
        self.named_monitors = {}
        for k, v in select_dict(_moai_, C._MONITORS_COLLECTION_).items():
            self.named_monitors[k] = Tensors(
                monitors, **v
            )
        ## Termination Criteria
        self.named_criteria = {}
        # for k in toolz.get_in(['criteria'], monitors) or {}:
        for k, v in select_dict(_moai_, C._CRITERIA_COLLECTION_).items():
            self.named_criteria[k] = Criteria(
                select(parameters, C._CRITERIA_), **v
            )
        #NOTE: up to here anything with optimizable parameters that can be selected
        #       should be created and available in `self`
        ## Optimizers & Parameters
        self.named_optimizers = OrderedDict()
        self.named_schedulers = defaultdict(dict) # {'step': {}, 'epoch': {}}
        # for k in toolz.get_in(["optimization", "optimizers", "named"], parameters) or {}:
        self.optimization_config = {
            '_optimizers_collection_': OmegaConf.to_container(select_conf(_moai_, C._OPTIMIZERS_COLLECTION_)),
            '_groups_': OmegaConf.to_container(select_conf(parameters, C._GROUPS_)),
            '_optimizers_': OmegaConf.to_container(select_conf(parameters, C._OPTIMIZERS_)),
            '_schedulers_collection_': OmegaConf.to_container(select_conf(_moai_, C._SCHEDULERS_COLLECTION_)),
            '_schedulers_': OmegaConf.to_container(select_conf(parameters, C._SCHEDULERS_)),
        }
        self.reset_optimization()        
        # Intializers
        self.named_initializers = defaultdict(list)         
        for k, v in select_dict(_moai_, f"{C._EXECUTION_INITIALIZE_}._{_moai_._action_}_").items():
            v = ensure_string_list(v)
            self.named_initializers[k] = [hyu.instantiate(parameters.initializers[i]) for i in v]
        ## Optimization Process & Monitoring
        self.process = OmegaConf.to_container(
            select(_moai_, C._EXECUTION_LIGHTNING_STEP_), resolve=True
        )
        self.monitor = OmegaConf.to_container(
            select(_moai_, C._EXECUTION_MONITORING_), resolve=True
        )
        self.schedule = deque(sorted(OmegaConf.to_container(
                select_conf(_moai_, C._EXECUTION_SCHEDULE_), resolve=True
            ), key=lambda item: item[C._EPOCH_],
        ))
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
        
    def reset_optimization(self) -> None:
        ## Optimizers
        for k, v in self.optimization_config['_optimizers_collection_'].items():
            groups = [self.optimization_config['_groups_'][g] for g in ensure_string_list(get(v, C._OPTIMIZER_GROUPS_))]
            optimizer = self.optimization_config['_optimizers_'][get(v, C._TYPE_)]
            config = OmegaConf.merge(optimizer, get_dict(v, C._PARAMS_))
            selected_params = list(hyu.instantiate(g)(self) for g in groups)
            self.named_optimizers[k] = hyu.instantiate(config, selected_params)
        ## Schedulers
        for k, v in self.optimization_config['_schedulers_collection_'].items():
            scheduler = self.optimization_config['_schedulers_'][get(v, C._TYPE_)]
            config = OmegaConf.merge(scheduler, get_dict(v, C._PARAMS_))
            interval = get(v, C._INTERVAL_) or C._EPOCH_ #NOTE: if missing defaults to `epoch`
            self.named_schedulers[interval][k] = hyu.instantiate(config, self.named_optimizers[get(v, C._OPTIMIZER_)])

    def setup_initializers(self) -> None: # call the initializers once at the beginning
        for init in self.named_initializers[C._SETUP_]:
            init(self)
    
    def batch_initializers(self) -> None: # call the initializers at the beginning of each batch
        for init in self.named_initializers[C._BATCH_]:
            init(self)
    
    def epoch_initializers(self) -> None: # call the initializers at the beginning of each epoch
        for init in self.named_initializers[C._EPOCH_]:
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
            a(self.named_flows, accessor(tensors))

    def predict_step(self,
        batch:  typing.Dict[str, torch.Tensor],
        batch_idx: int,
        dataset_idx: int=0,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        log.info(f"Predicting batch {batch_idx} ...")
        batch = benedict.benedict(batch, keyattr_enabled=False)        
        monitor = toolz.get_in([C._PREDICT_, C._BATCH_], self.monitor) or []
        for stage, proc in self.process[C._PREDICT_][C._BATCH_].items():
            steps = proc[C._FLOWS_]
            with torch.no_grad(): #TODO: probably this is not needed
                # for iter in range(iters): #NOTE: is this necessary?
                for step in steps:
                    batch = self.named_flows[step](batch)
                # predict step does 
                if monitor:
                    # Metrics monitoring used only for serve
                    for metric in toolz.get(C._METRICS_, monitor, None) or []: #TODO ADD _metrics_
                        self.named_metrics[metric](batch)
                    # Tensor monitoring for visualization & exporting
                    tensor_monitors = toolz.get(C._MONITORS_, monitor, None) or []
                    for tensor_monitor in tensor_monitors:
                        extras = {
                            'stage': 'predict',
                            'lightning_step': self.global_step,
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
            self.named_objectives[objective](tensors)
            loss = tensors[f"{C._MOAI_LOSSES_}.total"]
            is_first_batch_to_accumulate = index % self.trainer.accumulate_grad_batches == 0
            if self.trainer.accumulate_grad_batches == 1 or not is_first_batch_to_accumulate:
                call._call_callback_hooks(self.trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(self.trainer, "on_before_zero_grad", optimizer)
                call._call_lightning_module_hook(self.trainer, "optimizer_zero_grad", self.trainer.current_epoch, index, optimizer)
            call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
            self.optimization_step += 1            
            if monitor := toolz.get_in([C._FIT_, C._LIGHTNING_STEP_, stage], self.monitor):
                should_monitor = self.optimization_step % monitor.get('_frequency_', 1) == 0
                if (tensor_monitor_steps := get_list(monitor, C._MONITORS_)) and should_monitor:
                    with torch.no_grad():
                        for step in toolz.get_in([C._FLOWS_], monitor) or []:
                            self.named_flows[step](tensors)
                        extras = {
                            'lightning_step': self.global_step, 'epoch': self.current_epoch,
                            'optimization_step': self.optimization_step,
                            'batch_idx': batch_idx, 'stage': stage,
                        }
                        for step in tensor_monitor_steps:
                            self.named_monitors[step](tensors, extras)
            return loss
        batch = benedict.benedict(batch, keyattr_enabled=False)
        batch[C._MOAI_METRICS_] = {}
        #TODO: check for refresh optimizers each step
        for stage, proc in self.process[C._FIT_][C._BATCH_].items():
            flows = proc[C._FLOWS_]
            objective = proc.get(C._OBJECTIVE_, None)
            assign_params = proc.get(C._ASSIGN_, None)
            if optim := proc.get(C._OPTIMIZER_, None):
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
            current_closure = functools.partial(closure, batch, batch_idx, flows, stage, optimizer, objective)
            for iter in range(proc.get(C._ITERATIONS_, 1)):
                batch[C._MOAI_LOSSES_] = { 'raw': {}, 'weighted': {}, }
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
                            for flow in flows:
                                batch = self.named_flows[flow](batch)
                if iter_monitor_stage := toolz.get_in([C._FIT_, C._STAGE_, stage], self.monitor):
                    frequency = toolz.get(C._FREQUENCY_, iter_monitor_stage, 1)
                    should_monitor = iter % frequency == 0
                    if (iter_tensor_monitor := iter_monitor_stage.get(C._MONITORS_, None)) and should_monitor:
                        for step in toolz.get(C._FLOWS_, iter_monitor_stage, None) or []:
                            self.named_flows[step](batch)
                        for metric in toolz.get(C._METRICS_, iter_monitor_stage, None) or []:
                            self.named_metrics[metric](batch)
                        extras = {#TODO: step => 'lightning_step'
                            'lightning_step': self.global_step, 'epoch': self.current_epoch,
                            'batch_idx': batch_idx, 'stage': stage, 'iteration': iter
                        }
                        for step in iter_tensor_monitor:
                            self.named_monitors[step](batch, extras)
                        should_stop = False
                        for criterion in get_list(iter_monitor_stage, C._TERMINATION_):
                            if self.named_criteria[criterion](batch, extras):
                                should_stop = True
                                break
                        if should_stop:
                            log.info(f"Terminating {stage} @ {iter} with criterion [{criterion}] !")
                            break
            # call the copy params for initialization
            if assign_params is not None:
                frequency = assign_params.get(C._FREQUENCY_, 1) # default to each batch end
                assigners = []
                if batch_idx == 0: # get initializers only in the first batch                    
                    for i, o in assign_params.items():
                        if i == C._FREQUENCY_: #NOTE: refactor this, keys should be not coupled like this
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
        batch[C._MOAI_METRICS_] = {}
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
        batch[C._MOAI_METRICS_] = {}
        if not hasattr(self.data, 'val'):
            log.warning("Validation data missing. An empty validation set will be used.")
            return
        datasets = list(self.data.val.iterator.datasets.keys())
        monitor = toolz.get_in([C._VAL_, C._BATCH_], self.monitor)
        for stage, proc in (toolz.get_in([C._VAL_, C._BATCH_, datasets[dataloader_idx]], self.process, {}) or {}).items():
            steps = proc[C._FLOWS_]
            with torch.no_grad():
                for step in steps:
                    batch = self.named_flows[step](batch)
                for _, monitor_stage in monitor.items():
                    for metric in toolz.get(C._METRICS_, monitor_stage, None) or []:
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