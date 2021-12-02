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
from moai.parameters.optimization.optimizers import LBFGS as miLBFGS

import moai.utils.parsing.rtp as mirtp

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

__all__ = ['Optimizer']

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

class PerBatch(torch.nn.Identity, pytorch_lightning.Callback):
    def __init__(self):
        super(PerBatch, self).__init__()

    def on_train_batch_start(self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
        batch: typing.Dict[str, typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Dict[str, torch.Tensor]]],
        batch_idx: int,
        unused: typing.Optional[int] = 0,
    ) -> None:
        """Called when the train batch begins."""
        pl_module.initialize_parameters()
        if batch_idx > 0:
            trainer.accelerator.setup_optimizers(trainer)        
        if pl_module.mode == 'inference':
            with torch.no_grad():
                pl_module.preprocess(batch)
                pl_module(batch)
                pl_module.assign(batch)
            if pl_module.optimized_predictions:
                for key, values in pl_module.optimized_predictions.items():
                    for optim in filter(lambda o: o.name == key, trainer.accelerator.optimizers):
                        for v in values:
                            params = pl_module.predictions[v]
                            if isinstance(optim, torch.optim.LBFGS) or\
                                isinstance(optim, miLBFGS):
                                    optim._params.append(params)
                            else:
                                optim.param_groups.append({'params': params})
        pl_module.optimization_step = 0

from moai.monads.execution.cascade import _create_accessor

def _create_assigner(key: str) -> typing.Callable[[torch.nn.Module, torch.Tensor], None]:
    split = key.split('.')
    def _assign(m: torch.nn.Module, t: torch.Tensor, keys: typing.Sequence[str]):
         to_set = toolz.reduce(getattr, split, m)
         to_set.copy_(t)
    return functools.partial(_assign, keys=split)

class Optimizer(pytorch_lightning.LightningModule):
    def __init__(self, 
        inner:              omegaconf.DictConfig,
        configuration:      omegaconf.DictConfig,
        data:               omegaconf.DictConfig=None,
        parameters:         omegaconf.DictConfig=None,
        feedforward:        omegaconf.DictConfig=None,
        monads:             omegaconf.DictConfig=None,        
        supervision:        omegaconf.DictConfig=None,
        validation:         omegaconf.DictConfig=None,        
        visualization:      omegaconf.DictConfig=None,
        export:             omegaconf.DictConfig=None,
        hyperparameters:    typing.Union[omegaconf.DictConfig, typing.Mapping[str, typing.Any]]=None,
    ):
        super(Optimizer, self).__init__()
        self.model = hyu.instantiate(inner)
        self.fwds = []
        params = inspect.signature(self.model.forward).parameters
        model_in = list(zip(*[mirtp.force_list(configuration.io[prop]) for prop in params]))
        model_out = mirtp.split_as(mirtp.resolve_io_config(configuration.io['out']), model_in)
        self.res_fill = [mirtp.get_result_fillers(self.model, out) for out in model_out]
        get_filler = iter(self.res_fill)
        self.predictions = { }
        self.setters = []
        for keys in model_in:          
            accessors = [_create_accessor(k if isinstance(k, str) else k[0]) for k in keys]  
            self.fwds.append(lambda td,
                tk=keys,
                acc=accessors,
                args=params.keys(),
                model=self.model,
                filler=next(get_filler):
                    filler(td, model(**dict(zip(args,
                        list(acc[i](td) if type(k) is str
                            else list(td[j] for j in k)
                        for i, k in enumerate(tk))
                    )))) # list(td[k] if type(k) is str
            )
            self.setters.extend(toolz.concat(model_out))
        # self.mode = omegaconf.OmegaConf.select(configuration, 'mode', default='inference') #NOTE: update when omegaconf/hydra updates
        self.mode = omegaconf.OmegaConf.select(configuration, 'mode')
        if self.mode == 'inference':
            self.model = self.model.eval()
            self.model.requires_grad_(False)
        self.data = _assign_data(data)
        self.assigners = [(i, _create_assigner(o)) for i, o in configuration.assign.items()]
        self.optimized_predictions = configuration.optimize_predictions
        self.initializer = parameters.initialization if parameters is not None else None
        self.optimizer_configs = parameters.optimization.optimizers
        self.parameter_selectors = parameters.selectors
        self.scheduler_configs = parameters.optimization.schedulers or { }
        self.supervision = torch.nn.ModuleDict()
        self.params_optimizers = []
        self.stages = []
        log.info(f"Optimization with {len(parameters.optimization.process)} stages.")
        for stage, cfg in parameters.optimization.process.items():
            self.stages.append(stage)
            log.info(
                f"Setting up the '{stage}' stage using the " + str(cfg.optimizer or "same") + " optimizer, "
                f"optimizing the " + str(cfg.selectors or "same") + " parameters"
            )
            optimizer = cfg.optimizer
            iterations = cfg.iterations
            scheduler = cfg.scheduler
            selector = cfg.selectors
            self.params_optimizers.append((optimizer, stage, iterations, selector, scheduler))
            objective = cfg.objective #TODO: merge confs from supervision and objectives
            self.supervision[stage] = _create_supervision_block(
                omegaconf.OmegaConf.merge(supervision, objective)
            )
        self.validation = _create_validation_block(validation) #TODO: change this, "empty processing block" is confusing
        self.preprocess = _create_processing_block(feedforward, "preprocess", monads=monads)
        self.postprocess = _create_processing_block(feedforward, "postprocess", monads=monads)
        self.visualizer = _create_interval_block(visualization)
        self.exporter = _create_interval_block(export)        
        #NOTE: __NEEDED__ for loading checkpoint
        hparams = hyperparameters if hyperparameters is not None else { }
        hparams.update({'moai_version': miV})
        #NOTE: @PTL1.5 self.hparams =  hparams
        self.hparams.update(hparams)
        self.optimization_step = 0
        self.per_batch = PerBatch()

    def initialize_parameters(self) -> None:
        init = hyu.instantiate(self.initializer) if self.initializer else NoInit()
        init(self)

    def assign(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> None:
        for i, a in self.assigners:
            a(self, tensors[i])

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for f in self.fwds:
            f(tensors)
        for k in self.setters:
            self.predictions[k] = tensors[k]
        return tensors

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
        optimizer_idx:          int=0,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        batch['__moai__'] = { 'batch_index': batch_idx }
        batch['__moai__']['optimization_stage'] = self.stages[optimizer_idx]
        td = self.preprocess(batch)
        if self.mode == 'predict':
            td = self(td)
        td = self.postprocess(td)
        total_loss, losses = self.supervision[self.stages[optimizer_idx]](td)
        #TODO: should add loss maps as return type to be able to forward them for visualization
        losses = toolz.keymap(lambda k: f"train_{k}", losses)
        losses.update({'total_loss': total_loss})
        self.log_dict(losses, prog_bar=False, logger=True)
        self.optimization_step += 1
        return { 'loss': total_loss, 'tensors': td }

    def training_step_end(self, 
        train_outputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]
    ) -> None:
        train_outputs['tensors']['__moai__']['optimization_step'] = self.optimization_step
        if (self.optimization_step + 1) and (self.optimization_step % self.visualizer.interval == 0):
            self.visualizer(train_outputs['tensors'])
        if (self.optimization_step + 1) and (self.optimization_step % self.exporter.interval == 0):
            self.exporter(train_outputs['tensors'])
        return train_outputs['loss']

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        log.info(f"Configuring optimizers and schedulers")                
        optimizers, schedulers = [], []        
        for optimizer, name, iterations, params, schedule in self.params_optimizers:
            if optimizer is None and params is None:
                optimizers.append(optimizers[-1])
                schedulers.append(schedulers[-1])
                getattr(optimizers[-1], 'iterations').append(iterations)
                getattr(optimizers[-1], 'name').append(name)
            else:
                parameters = hyu.instantiate(self.parameter_selectors[params])(self) if isinstance(params, str) else\
                    list(toolz.concat(hyu.instantiate(self.parameter_selectors[p])(self) for p in params))
                optimizers.append(_create_optimization_block(
                    self.optimizer_configs[optimizer], parameters
                ))
                setattr(optimizers[-1], 'iterations', [iterations])
                setattr(optimizers[-1], 'name', [name])
                schedulers.append(_create_scheduling_block(
                    self.scheduler_configs.get(schedule, None), [optimizers[-1]]
                ))
        return (
            optimizers,
            list(map(lambda s: s.schedulers[0] if isinstance(s, NoScheduling) else s, schedulers))
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        #NOTE: we are in train mode as we may need to optimize weights,
        # but semantically its test time so we use the test set
        if hasattr(self.data.test.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.test.iterator._target_.split('.')[-1]}) test set data iterator")
            test_iterator = hyu.instantiate(self.data.test.iterator)
        else:
            test_iterator = Indexed(
                self.data.test.iterator.datasets,
                self.data.test.iterator.augmentation if hasattr(self.data.test.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.test, 'loader'):
            log.error("Test data loader missing. Please add a data loader (i.e. \'- data/test/loader: torch\') entry in the configuration.")
        else:
            test_loader = hyu.instantiate(self.data.test.loader, test_iterator, 
                shuffle=False, batch_size=1
            )
        return test_loader