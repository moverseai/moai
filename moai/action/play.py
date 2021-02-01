import moai.engine.lightning.play as miplay

import os
import sys
import hydra
import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

def play(cfg):
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    engine = hydra.utils.instantiate(cfg.engine)
    model = miplay.Presenter(
        **(cfg.model if hasattr(cfg, "model") else {"monads": None}) ,
        data=cfg.data,
        visualization=cfg.visualization
    )
    player = hydra.utils.instantiate(cfg.player)
    log.info("Dataset showcase started.")
    player.play(model)
    log.info("Dataset showcase completed.")

if __name__ == "__main__":  
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    config_filename = sys.argv.pop(1) #TODO: argparser integration?   
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")
    play = hydra.main(config_path="conf", config_name=config_filename)(play)
    play()