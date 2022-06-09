from ts.torch_handler.base_handler import BaseHandler
from hydra.experimental import (
    compose,
    initialize,
)
from omegaconf.omegaconf import OmegaConf

import hydra.utils as hyu
import toolz
import os
import torch
import zipfile
import yaml
import logging
import typing

log = logging.getLogger(__name__)

__all__ = ['OptimizerServer']

class OptimizerServer(BaseHandler):
    def __init__(self) -> None:
        super().__init__()

    def _extract_files(self):
        with zipfile.ZipFile('conf.zip', 'r') as conf_zip:
            conf_zip.extractall('.')
        with zipfile.ZipFile('src.zip', 'r') as src_zip:
            src_zip.extractall('.')

    def _get_overrides(self):
        if os.path.exists('overrides.yaml'):
            with open('overrides.yaml') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        else:
            return []

    def initialize(self, context):        
        properties = context.system_properties
        # self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            gpu_id = properties.get("gpu_id")
            if gpu_id is not None:
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cuda")
        # self.device = torch.device(
        #     self.map_location + ":" + str(properties.get("gpu_id"))
        #     if torch.cuda.is_available() and properties.get("gpu_id") is not None
        #     else self.map_location
        # )
        log.info(f"Model set to run on a [{self.device}] device")
        #NOTE: IMPORTANT!!!! DEBUG WHILE TRAINING ONLY !!!
        # log.warning(f"IMPORTANT: Model explicitly set to CPU mode for debugging purposes! (was {self.device}).")
        # self.device = torch.device('cpu')        
        #NOTE: IMPORTANT!!!! DEBUG WHILE TRAINING ONLY !!!
        main_conf = context.manifest['model']['modelName'].replace('_', '/')
        log.info(f"Loading the {main_conf} endpoint.")
        self._extract_files()        
        try:
            with initialize(config_path="conf", job_name=main_conf):
                cfg = compose(
                    config_name=main_conf, 
                    overrides=self._get_overrides()
                )
                self.optimizer = hyu.instantiate(cfg.model)
                self.engine = hyu.instantiate(cfg.engine)
        except Exception as e:
            log.error(f"An error has occured while loading the model:\n{e}")
        self.optimizer = self.optimizer.to(self.device)
        self.optimizer.eval()
        self.initialized = True
        log.info(f"Model ({type(self.optimizer.model)}) loaded successfully.")
        self.gradient_tolerance = cfg.fitter.gradient_tolerance
        self.relative_tolerance = cfg.fitter.relative_tolerance
        try:
            handler_overrides = None
            if os.path.exists('handler_overrides.yaml'):
                with open('handler_overrides.yaml') as f:
                    handler_overrides = yaml.load(f, Loader=yaml.FullLoader)
                    log.debug(f"Loaded handler overrides:\n{handler_overrides}")
            with initialize(config_path="conf", job_name=f"{main_conf}_preprocess_handlers"):
                cfg = compose(config_name="../pre")
                if handler_overrides is not None and 'preprocess' in handler_overrides.get('handlers', {}):
                    cfg = OmegaConf.merge(cfg, handler_overrides)
                    log.debug(f"Merged handler overrides:\n{cfg}")
                self.preproc = {k: hyu.instantiate(h) for k, h in (toolz.get_in(['handlers', 'preprocess'], cfg) or {}).items()}
            with initialize(config_path="conf", job_name=f"{main_conf}_postprocess_handlers"):
                cfg = compose(config_name="../post")
                if handler_overrides is not None and 'postprocess' in handler_overrides.get('handlers', {}):
                    cfg = OmegaConf.merge(cfg, handler_overrides)
                    log.debug(f"Merged handler overrides:\n{cfg}")
                self.postproc = {k: hyu.instantiate(h) for k, h in (toolz.get_in(['handlers', 'postprocess'], cfg) or {}).items()}
        except Exception as e:
            log.error(f"An error has occured while loading the handlers:\n{e}")

    def preprocess(self, 
        data:   typing.Mapping[str, typing.Any],
    ) -> typing.Dict[str, torch.Tensor]:
        log.debug(f"Preprocessing input:\n{data}")
        tensors = { '__moai__': { 'json': data } }
        body = data[0].get('body') or data[0].get('raw')
        for k, p in self.preproc.items():
            tensors = toolz.merge(tensors, p(body, self.device))
        log.debug(f"Tensors: {tensors.keys()}")
        return tensors

    def relative_check(self, 
        prev: torch.Tensor, 
        current: torch.Tensor
    ) -> float:
        relative_change = (prev - current) / max([prev.abs(), current.abs(), 1.0])
        return relative_change <= self.relative_tolerance

    def gradient_check(self, 
        param_groups: typing.Sequence[typing.Dict[str, torch.nn.Parameter]]
    ) -> bool:
        return all(
            p.grad.view(-1).max().abs().item() < self.gradient_tolerance 
            for p in toolz.concat((g['params'] for g in param_groups)) 
            if p.grad is not None
        )

    def is_any_param_nan(self, optimizer: torch.optim.Optimizer) -> bool:
        for pg in optimizer.param_groups:
                for p in pg['params']:
                    if not torch.all(torch.isfinite(p)):
                        return True
        return

    def inference(self, 
        data:       typing.Mapping[str, torch.Tensor],
    ):        
        self.last_loss = None
        self.optimizer.initialize_parameters()
        optimizers, schedulers = self.optimizer.configure_optimizers()
        iters = list(toolz.mapcat(lambda o: o.iterations, toolz.unique(optimizers)))
        stages = list(toolz.mapcat(lambda o: o.name, toolz.unique(optimizers)))
        if self.optimizer.mode == 'inference':                
            with torch.no_grad():
                self.optimizer.preprocess(data)
                self.optimizer(data)
                self.optimizer.initialize(data)            
        for i, (optim, iters, stage, sched) in enumerate(zip(
            optimizers, iters, stages, schedulers
        )):
            log.info(f"Optimizing stage: {stage} for {iters} iterations")
            for p in self.optimizer.parameters():
                p.requires_grad_(False)
            for pg in optim.param_groups:
                for p in pg['params']:
                    p.requires_grad_(True)
            def closure():
                self.optimizer.optimizer_zero_grad(
                    epoch=0, batch_idx=0,
                    optimizer=optim, optimizer_idx=i
                )
                self.loss = self.optimizer.training_step(batch=data, 
                    batch_idx=0, optimizer_idx=i
                )['loss']
                self.loss.backward()
                self.optimizer.optimization_step += 1
                data['__moai__']['optimization_step'] = self.optimizer.optimization_step
                return self.loss
            for j in range(iters):
                optim.step(closure=closure)                            
                current_loss = self.loss
                if hasattr(optim, 'assign'):
                    with torch.no_grad():
                        optim.assign(data)
                if (self.last_loss is not None and self.relative_check(
                    self.last_loss, current_loss
                )) or self.gradient_check(optim.param_groups)\
                    or not torch.isfinite(current_loss)\
                    or self.is_any_param_nan(optim):
                        log.warning(f"Optimization stage '{stage}' stopped at iteration {j}/{iters}.")
                        break
                self.last_loss = current_loss
            sched.step()
            self.last_loss = None
            self.optimizer.optimization_step = 0
        return data 
       
    def postprocess(self,
        data: typing.Mapping[str, torch.Tensor]
    ) -> typing.Sequence[typing.Any]:
        log.debug(f"Postprocessing outputs:\n{data['__moai__']}")
        outs = []
        for k, p in self.postproc.items():
            res = p(data, data['__moai__']['json'])
            if len(outs) == 0:
                outs = res
            else:                
                for o, r in zip(outs, res):
                    o = toolz.merge(o, r)
        return outs

    # def handle(self, data, context):
    #     return super(data, context)