import hydra
import sys
import omegaconf.omegaconf
import logging
import typing

import os
import re

from moai.engine.callbacks.model import ModelCallbacks

log = logging.getLogger(__name__)

def assign(cfg: omegaconf.DictConfig, attr: str) -> typing.Union[typing.Any,typing.Any]:
    return getattr(cfg, attr) if hasattr(cfg, attr) else None

def run(cfg):
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    with open("config.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    engine = hydra.utils.instantiate(cfg.engine)
    model = hydra.utils.instantiate(cfg.model,
        data=cfg.data,
        # visualization=assign(cfg, "visualization"),
        # export=assign(cfg, "export"),
        monitoring=assign(cfg, "monitoring"), #NOTE: should this be added to `model.` ?
        stopping=assign(cfg, "stopping"), #NOTE: should this be added to `model.` ?
        _recursive_=False
    )
    for name, remodel in (assign(cfg ,"remodel") or {}).items():
        hydra.utils.instantiate(remodel)(model)
    #NOTE: why do we need to pass model a
    fitter = hydra.utils.instantiate(cfg.fitter, 
        logging=assign(cfg, "logging"),
        model_callbacks=ModelCallbacks(model=model),
        _recursive_=False
    )
    # select the appropriate action to run
    if cfg.action == "test":
        log.info("Evaluation started.")
        #NOTE: what is the difference between `predict` and `test`?
        ckpt_path = cfg.get('ckpt_path', None)
        #TODO: check how can we have different intialisation for test, train, etc.
        if ckpt_path is None:
            log.warning("No checkpoint path provided, test will be done with custom initialisation if present.")
        fitter.test(model, ckpt_path=ckpt_path)
        log.info("Evaluation finished.")
    elif cfg.action == "predict":
        log.info("Prediction started.")
        ckpt_path = cfg.get('ckpt_path', None)
        #TODO: check how can we have different intialisation for test, train, etc.
        if ckpt_path is None:
            log.warning("No checkpoint path provided, test will be done with custom initialisation if present.")
        fitter.predict(model, ckpt_path=ckpt_path)
        log.info("Prediction finished.")
    elif cfg.action == "fit":
        log.info("Fitting started.")
        fitter.run(model)
        log.info("Fitting completed.")
    elif cfg.action == "resume":
        pass
        #TODO: implement resume

    else:
        log.error(f"Action {cfg.action} not recognized.")

if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")
    run = hydra.main(config_path="conf", config_name=config_filename, version_base='1.3')(run)
    run()