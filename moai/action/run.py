import hydra
from omegaconf.omegaconf import OmegaConf, DictConfig
import logging
import typing

from moai.utils.funcs import select
from moai.engine import Engine
from moai.engine.callbacks.model import ModelCallbacks
from moai.core.execution.constants import Constants

log = logging.getLogger(__name__)

# def _assign(cfg: omegaconf.DictConfig, attr: str) -> typing.Union[typing.Any,typing.Any]:
#     return getattr(cfg, attr) if hasattr(cfg, attr) else None

def _specialize_config(cfg: DictConfig) -> None:
    #TODO: remove unused keys for metrics etc. inside the named collections/flows
    # match cfg.action:
    #     case 'fit':
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.test'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').test
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.predict'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').predict
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.test'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').test
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.predict'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').predict
    #     case 'test':
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.fit'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').fit
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.val'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').val
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.predict'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').predict
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.fit'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').fit
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.val'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').val
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.predict'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').predict
    #     case 'predict':
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.fit'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').fit
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.val'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').val
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.test'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').test
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.fit'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').fit
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.val'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').val
    #         if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.test'):
    #             del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').test
    #     case _:
    #         log.error(f"Erroneous `action` [{cfg.action}] specified on `run` mode.")
    if cfg.action == 'fit':
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.test'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').test
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.predict'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').predict
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.test'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').test
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.predict'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').predict
        if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.test'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}').test
        if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.predict'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}').predict
        # if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.fit'):
        #     cfg._moai_._init_ = OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.fit')

    elif cfg.action == 'test':
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.fit'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').fit
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.val'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').val
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.predict'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').predict
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.fit'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').fit
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.val'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').val
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.predict'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').predict
        if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.fit'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}').fit
        if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.predict'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}').predict
    elif cfg.action == 'predict':
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.fit'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').fit
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.val'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').val
        if OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}.test'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_EXECUTION_}').test
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.fit'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').fit
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.val'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').val
        if OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}.test'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_MONITORING_}').test
        if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.test'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}').test
        if OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}.fit'):
            del OmegaConf.select(cfg, f'{Constants._MOAI_INITIALIZE_}').fit
    else:
        log.error(f"Erroneous `action` [{cfg.action}] specified on `run` mode.")

def run(cfg: DictConfig):
    _specialize_config(cfg)
    hydra.utils.log.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    with open("config.yaml", 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    # engine = hydra.utils.instantiate(cfg.engine)
    engine = Engine(cfg.engine.modules) #NOTE: ctor before model/runner
    OmegaConf.set_struct(cfg, False)
    cfg._moai_._action_ = cfg.action
    OmegaConf.set_struct(cfg, True)
    model = hydra.utils.instantiate(cfg.model,
        data=cfg.data,
        # visualization=assign(cfg, "visualization"),
        # export=assign(cfg, "export"),
        # monitoring=assign(cfg, "monitoring"), #NOTE: should this be added to `model.` ?
        # stopping=assign(cfg, "stopping"), #NOTE: should this be added to `model.` ?
        _moai_=select(cfg, Constants._MOAI_),
        _recursive_=False
    )
    # for name, remodel in (assign(cfg ,"remodel") or {}).items():
    #     hydra.utils.instantiate(remodel)(model)
    #NOTE: why do we need to pass model a
    runner = hydra.utils.instantiate(cfg.engine.runner, 
        #logging=assign(cfg, "logging"),
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
        runner.test(model, ckpt_path=ckpt_path)
        log.info("Evaluation finished.")
    elif cfg.action == "predict":
        log.info("Prediction started.")
        ckpt_path = cfg.get('ckpt_path', None)
        #TODO: check how can we have different intialisation for test, train, etc.
        if ckpt_path is None:
            log.warning("No checkpoint path provided, test will be done with custom initialisation if present.")
        runner.predict(model, ckpt_path=ckpt_path)
        log.info("Prediction finished.")
    elif cfg.action == "fit":
        log.info("Fitting started.")
        runner.run(model)
        log.info("Fitting completed.")
    elif cfg.action == "resume":
        pass
        #TODO: implement resume

    else:
        log.error(f"Action {cfg.action} not recognized.")

# if __name__ == "__main__":
#     # os.environ['HYDRA_FULL_ERROR'] = '1'
#     config_filename = sys.argv.pop(1) #TODO: argparser integration?
#     sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")
#     run = hydra.main(config_path="conf", config_name=config_filename, version_base='1.3')(run)
#     run()