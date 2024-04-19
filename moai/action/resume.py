from moai import __version__ as miV

import hydra
import sys
import omegaconf.omegaconf
import logging
import typing
from omegaconf import OmegaConf
import re
import os
import hydra.utils as hyu
from moai.engine.callbacks import ModelCallbacks


log = logging.getLogger(__name__)

def get_conf(path: str) -> omegaconf.OmegaConf:
    if os.path.splitext(path)[1] != '.yaml':
        nominal_config_path = os.path.join(path, 'config.yaml')
        if os.path.exists(nominal_config_path):
            path = nominal_config_path
        elif os.path.basename(path) != '.hydra':
            path = os.path.join(path, '.hydra') #TODO: merge with overrides
            path = os.path.join(path, 'config.yaml')
    return path

def assign(cfg: omegaconf.DictConfig, attr: str) -> typing.Union[typing.Any,typing.Any]:
    return getattr(cfg, attr) if hasattr(cfg, attr) else None

@hydra.main(config_name="tools/resume.yaml", config_path="conf", version_base='1.3')
def resume(cfg):
    #find last checkpoint
    regex = re.compile('(last.*ckpt)')
    last_ckpt_path = None
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if regex.match(file):
                last_ckpt_path = os.path.join(root,file)
                break
    if not last_ckpt_path:
        log.error(f"The provided experiment path {cfg.experiment.path} does not contain a last checkpoint file.")
    
    resume_callback = hyu.instantiate(
        {
            "_target_": "moai.parameters.initialization.lightning.Continued",
            "filename": last_ckpt_path
        })
    config_file = get_conf(os.getcwd())
    with open(config_file, 'r') as f:
        init_cfg = OmegaConf.load(f)
    omegaconf.OmegaConf.set_struct(init_cfg, True)
    cfg = OmegaConf.merge(init_cfg,cfg) # to change param values
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    engine = hydra.utils.instantiate(cfg.engine)
    model = hydra.utils.instantiate(cfg.model,
        data=cfg.data,
        visualization=assign(cfg, "visualization"),
        export=assign(cfg, "export"),
    )
    model.hparams.update(omegaconf.OmegaConf.to_container(cfg, resolve=True))
    model.hparams['__moai__'] = { 'version': miV }
    for name, remodel in (assign(cfg, "remodel") or {}).items():
        hydra.utils.instantiate(remodel)(model)
    # model_callbacks = ModelCallbacks(model=model)
    # model_callbacks += [resume_callback]
    #NOTE: check https://github.com/Lightning-AI/pytorch-lightning/issues/12819
    # resume training is now supported directly in PTL 2.0
    trainer = hydra.utils.instantiate(cfg.trainer, 
        logging=assign(cfg, "logging"),
        # model_callbacks=model_callbacks,
    )
    log.info("Training started.")
    trainer.run(model, resume_from_checkpoint=last_ckpt_path)
    log.info("Training completed.")

if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")    
    resume = hydra.main(config_path="conf", config_name=config_filename)(resume)
    resume()