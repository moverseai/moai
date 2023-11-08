from ts.torch_handler.base_handler import BaseHandler
from hydra.experimental import (
    compose,
    initialize,
)
from omegaconf.omegaconf import OmegaConf
import omegaconf

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

def _find_all_targets(d, key="_target_", previous_key=None):
        t_dict = {}
        if isinstance(d, str):
            pass  # Strings are ignored, as they cannot contain a nested _target_ key
        elif isinstance(d, omegaconf.dictconfig.DictConfig):
            for k, v in d.items():
                if k == key:
                    # If the _target_ key is found, update the t_dict with the current dictionary
                    t_dict.update({previous_key: d})
                else:
                    # Recurse into the dictionary
                    child_dict = _find_all_targets(v, key, previous_key=k)
                    if child_dict:  # If the recursion found a _target_, update t_dict
                        t_dict.update(child_dict)
                    elif previous_key is not None and previous_key not in t_dict:  # If no _target_ found, add an empty dict
                        t_dict[previous_key] = {}
        elif isinstance(d, omegaconf.listconfig.ListConfig):
            pass # Lists are ignored, as they cannot contain a nested _target_ key
        else:
            if previous_key is not None:  # If no _target_ found, add an empty dict
                t_dict[previous_key] = {}
        return t_dict

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
        # self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and not 'FORCE_CPU' in os.environ:
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
                self.model = hyu.instantiate(cfg.model)
                self.engine = hyu.instantiate(cfg.engine)
                if hasattr(cfg, 'remodel'):
                    log.info(f"Remodeling...")
                    for k, v in cfg.remodel.items():
                        hyu.instantiate(v)(self.model)
        except Exception as e:
            log.error(f"An error has occured while loading the model:\n{e}")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        log.info(f"Model ({type(self.model.model if hasattr(self.model, 'model') else self.model)}) loaded successfully.")
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
                # self.preproc = {k: hyu.instantiate(h) for k, h in (toolz.get_in(['handlers', 'preprocess'], cfg) or {}).items()}
                self.preproc = {k: hyu.instantiate(h) for k, h in _find_all_targets(toolz.get_in(['handlers', 'preprocess'], cfg)).items()}
            with initialize(config_path="conf", job_name=f"{main_conf}_postprocess_handlers"):
                cfg = compose(config_name="../post")
                if handler_overrides is not None and 'postprocess' in handler_overrides.get('handlers', {}):
                    cfg = OmegaConf.merge(cfg, {'handlers': {'postprocess': handler_overrides['handlers']['postprocess']}})
                    log.debug(f"Merged handler overrides:\n{cfg}")
                # self.postproc = {k: hyu.instantiate(h) for k, h in (toolz.get_in(['handlers', 'postprocess'], cfg) or {}).items()}
                self.postproc = {k: hyu.instantiate(h) for k, h in _find_all_targets(toolz.get_in(['handlers', 'postprocess'], cfg)).items()}
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
        for k, v in metrics.items(): # self.context is set in base handler's handle method            
            self.context.metrics.add_metric(name=k, value=float(v.detach().cpu().numpy()), unit='value')
        return tensors
       
    def postprocess(self,
        data: typing.Mapping[str, torch.Tensor]
    ) -> typing.Sequence[typing.Any]:
        log.debug(f"Postprocessing outputs:\n{data['__moai__']}")
        outs = [] #TODO: corner case with no postproc crashes, fix it
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