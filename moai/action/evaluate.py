import hydra
import sys
import omegaconf.omegaconf
import logging
import typing
import tablib

from moai.engine.callbacks import ModelCallbacks

log = logging.getLogger(__name__)

def assign(cfg: omegaconf.DictConfig, attr: str) -> typing.Union[typing.Any, typing.Any]:
    return getattr(cfg, attr) if hasattr(cfg, attr) else None

def evaluate(cfg):
    # hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_container(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        # f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
        f.write(omegaconf.OmegaConf.to_yaml(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=False))
    with open("config.yaml", 'w') as f:
        # f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
        f.write(omegaconf.OmegaConf.to_yaml(omegaconf.OmegaConf.to_container(cfg, resolve=False), sort_keys=False))
    engine = hydra.utils.instantiate(cfg.engine)
    model = hydra.utils.instantiate(cfg.model,
        data=cfg.data,
        visualization=assign(cfg, "visualization"),
        export=assign(cfg, "export"),    
    )
    for name, remodel in (assign(cfg, "remodel") or {}).items():
        hydra.utils.instantiate(remodel)(model)
    model.initialize_parameters()
    tester = hydra.utils.instantiate(cfg.tester, 
        logging=assign(cfg, "logging"),
        model_callbacks=ModelCallbacks(model=model),
    )
    log.info("Evaluation started.")
    results = tester.run(model)[0] #TODO: need to support other ways of logging average results
   # ds = tablib.Dataset(results.values(), headers=results.keys())
    for dataset in results.keys():
        ds = tablib.Dataset([v.detach().cpu().numpy() for v in results[dataset].values()], headers=results[dataset].keys())
        with open(cfg.experiment.name +f'_{dataset}_test_average.csv', 'a', newline='') as f:
            f.write(ds.export('csv'))
    log.info("Evaluation completed.")

if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")    
    evaluate = hydra.main(config_path="conf", config_name=config_filename)(evaluate)
    evaluate()