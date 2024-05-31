from omegaconf.omegaconf import OmegaConf, DictConfig
from moai.utils.funcs import select
from moai.engine import Engine
from moai.engine.callbacks.model import ModelCallbacks
from moai.core.execution.constants import Constants as C

import hydra
import logging

log = logging.getLogger(__name__)

def _specialize_config(cfg: DictConfig) -> None:
    #TODO: remove unused keys for metrics etc. inside the named collections/flows
    # match cfg.action:
    #     case 'fit':
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.test'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').test
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.predict'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').predict
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.test'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').test
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.predict'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').predict
    #     case 'test':
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.fit'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').fit
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.val'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').val
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.predict'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').predict
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.fit'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').fit
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.val'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').val
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.predict'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').predict
    #     case 'predict':
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.fit'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').fit
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.val'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').val
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}.test'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}').test
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.fit'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').fit
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.val'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').val
    #         if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}.test'):
    #             del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}').test
    #     case _:
    #         log.error(f"Erroneous `action` [{cfg.action}] specified on `run` mode.")
    if cfg.action == C._ACTION_FIT_:
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._test_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._test_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._predict_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._predict_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._test_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._test_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._predict_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._predict_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}._test_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}')._test_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}._predict_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}')._predict_        

    elif cfg.action == C._ACTION_TEST_:
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._fit_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._fit_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._val_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._val_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._predict_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._predict_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._fit_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._fit_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._val_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._val_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._predict_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._predict_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}._fit_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}')._fit_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}._predict_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}')._predict_
    
    elif cfg.action == C._ACTION_PREDICT_:
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._fit_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._fit_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._val_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._val_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}._test_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_LIGHTNING_STEP_}')._test_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._fit_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._fit_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._val_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._val_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}._test_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_MONITORING_}')._test_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}._test_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}')._test_
        if OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}._fit_'):
            del OmegaConf.select(cfg, f'{C._MOAI_EXECUTION_INITIALIZE_}')._fit_
    else:
        log.error(f"Erroneous `action` [{cfg.action}] specified on `run` mode.")

def run(cfg: DictConfig):
    _specialize_config(cfg)
    hydra.utils.log.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    with open("config.yaml", 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    engine = Engine(cfg.engine.modules) #NOTE: ctor before model/runner
    OmegaConf.set_struct(cfg, False)
    cfg._moai_._action_ = cfg.action
    OmegaConf.set_struct(cfg, True)
    model = hydra.utils.instantiate(cfg.model, data=cfg.data,
        _moai_=select(cfg, C._MOAI_), _recursive_=False
    )
    runner = hydra.utils.instantiate(cfg.engine.runner,         
        model_callbacks=ModelCallbacks(model=model),
        _recursive_=False
    )
    # select the appropriate action to run
    if cfg.action == C._ACTION_TEST_:
        log.info("Evaluation started.")
        ckpt_path = cfg.get('ckpt_path', None)
        #TODO: check how can we have different intialisation for test, train, etc.
        if ckpt_path is None:
            log.warning("No checkpoint path provided, test will be done with custom initialisation if present.")
        runner.test(model, ckpt_path=ckpt_path)
        log.info("Evaluation finished.")
    elif cfg.action == C._ACTION_PREDICT_:
        log.info("Prediction started.")
        ckpt_path = cfg.get('ckpt_path', None)
        #TODO: check how can we have different intialisation for test, train, etc.
        if ckpt_path is None:
            log.warning("No checkpoint path provided, test will be done with custom initialisation if present.")
        runner.predict(model, ckpt_path=ckpt_path)
        log.info("Prediction finished.")
    elif cfg.action == C._ACTION_FIT_:
        log.info("Fitting started.")
        runner.run(model)
        log.info("Fitting completed.")
    elif cfg.action == C._ACTION_RESUME_:
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