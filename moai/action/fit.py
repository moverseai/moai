import hydra
import sys
import omegaconf.omegaconf
import logging
import typing

from moai.engine.callbacks import ModelCallbacks

log = logging.getLogger(__name__)

def assign(cfg: omegaconf.DictConfig, attr: str) -> typing.Union[typing.Any,typing.Any]:
    return getattr(cfg, attr) if hasattr(cfg, attr) else None

def fit(cfg):
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    with open("config.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    engine = hydra.utils.instantiate(cfg.engine)
    model = hydra.utils.instantiate(cfg.model,
        data=cfg.data,
        visualization=assign(cfg, "visualization"),
        export=assign(cfg, "export"),    
    )
    for name, remodel in (assign(cfg ,"remodel") or {}).items():
        hydra.utils.instantiate(remodel)(model)
    fitter = hydra.utils.instantiate(cfg.fitter, 
        logging=assign(cfg, "logging"),
        model_callbacks=ModelCallbacks(model=model),
    )
    log.info("Fitting started.")     
    fitter.run(model)
    log.info("Fitting completed.")

if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")    
    fit = hydra.main(config_path="conf", config_name=config_filename)(fit)
    fit()