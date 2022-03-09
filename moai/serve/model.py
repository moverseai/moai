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

__all__ = ['ModelServer']

class ModelServer(BaseHandler):
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
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        #NOTE: IMPORTANT!!!! DEBUG WHILE TRAINING ONLY !!!
        self.device = torch.device('cpu')
        log.warning("IMPORTANT: Model explicitly set to CPU mode for debugging purposes!")
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
                self.model = hyu.instantiate(cfg.model)
        except Exception as e:
            log.error(f"An error has occured while loading the model:\n{e}")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        log.info(f"Model ({type(self.model.inner if hasattr(self.model, 'inner') else self.model)}) loaded successfully.")        
        try:
            handler_overrides = None
            if os.path.exists('handler_overrides.yaml'):
                with open('handler_overrides.yaml') as f:
                    handler_overrides = yaml.load(f, Loader=yaml.FullLoader)
                    log.debug(f"Loaded handler overrides:\n{handler_overrides}")
            with initialize(config_path="conf", job_name=f"{main_conf}_preprocess_handlers"):
                cfg = compose(config_name="../pre")
                if handler_overrides is not None and 'preprocess' in handler_overrides.get('handlers', {}):
                    cfg = OmegaConf.merge(cfg, {'handlers': {'preprocess': handler_overrides['handlers']['preprocess']}})
                    log.debug(f"Merged handler overrides:\n{cfg}")
                self.preproc = {k: hyu.instantiate(h) for k, h in (toolz.get_in(['handlers', 'preprocess'], cfg) or {}).items()}
            with initialize(config_path="conf", job_name=f"{main_conf}_postprocess_handlers"):
                cfg = compose(config_name="../post")
                if handler_overrides is not None and 'postprocess' in handler_overrides.get('handlers', {}):
                    cfg = OmegaConf.merge(cfg, {'handlers': {'postprocess': handler_overrides['handlers']['postprocess']}})
                    log.debug(f"Merged handler overrides:\n{cfg}")
                self.postproc = {k: hyu.instantiate(h) for k, h in (toolz.get_in(['handlers', 'postprocess'], cfg) or {}).items()}
        except Exception as e:
            log.error(f"An error has occured while loading the handlers:\n{e}")
        self.model.initialize_parameters()

    def preprocess(self, 
        data:   typing.Mapping[str, typing.Any],
    ) -> typing.Dict[str, torch.Tensor]:
        log.debug(f"Preprocessing input:\n{data}")
        tensors = { '__moai__': { 'json': data } }
        #TODO: need to check batched inference
        body = data[0].get('body') or data[0].get('raw')
        for k, p in self.preproc.items():
            tensors = toolz.merge(tensors, p(body, self.device))
        log.debug(f"Tensors: {tensors.keys()}")
        return tensors
    
    def inference(self, 
        data:       typing.Mapping[str, torch.Tensor],
    ):
        metrics, tensors = self.model.test_step(data, batch_nb=0)
        return tensors
       
    def postprocess(self,
        data: typing.Mapping[str, torch.Tensor]
    ) -> typing.Sequence[typing.Any]:
        outs = []
        for k, p in self.postproc.items():
            outs.append(p(data, data['__moai__']['json']))
        return outs

    # def handle(self, data, context):
    #     return super(data, context)