try:
    from moai.action.train import train
    from moai.action.evaluate import evaluate
    from moai.action.play import play
    from moai.action.diff import diff
    from moai.action.plot import plot
except:
    from action.train import train
    from action.evaluate import evaluate
    from action.play import play
    from action.diff import diff
    from action.plot import plot

import omegaconf.omegaconf
import hydra
import logging
import sys
import os
import pkgutil

from rich.traceback import install
install(width=120, extra_lines=5, theme=None,
    word_wrap=True, show_locals=False, indent_guides=True,    
)

ERROR_FORMAT = "%(levelname)s at %(asctime)s in %(funcName)s in %(filename) at line %(lineno)d: %(message)s"
DEBUG_FORMAT = "%(lineno)d in %(filename)s at %(asctime)s: %(message)s"
# FORMAT = "[%(asctime)s][%(filename)s][%(levelname)s] - %(message)s"
FORMAT = "[%(levelname)s] - %(message)s"

logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger('moai')

def debug(cfg):
    log.info(cfg)

__MODES__ = {
    'train': train,
    'evaluate': evaluate,
    'play': play,
    'diff': diff,
    'plot': plot,
    'debug': debug,
}

__MIN_ARGS_COUNT__ = {
    'train': 2,
    'evaluate': 2,
    'play': 2,
    'diff': 1,
    'plot': 1,
    'debug': 2,
}

def run(cfg: omegaconf.DictConfig):
    __MODES__[cfg.mode](cfg)

def moai():    
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    mode = sys.argv.pop(1)
    if mode not in __MODES__:
        log.error(f"Wrong moai execution mode ({mode}), supported modes are: {list(__MODES__.keys())}.")
        sys.exit(-1)
    min_args = __MIN_ARGS_COUNT__[mode]
    if len(sys.argv) < min_args:
        log.error(f"Insufficient arguments provided for moai. Calling should specify the mode and config file: \'moai MODE CONFIG\'.")    
    config = sys.argv.pop(1) if min_args > 1 else f"tools/{mode}.yaml" 
    other_args = []
    if min_args > 1 or mode == 'plot':
        output_dir = "hydra.run.dir=actions/" + mode + "/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}"
        other_args.append(output_dir)
    else:
        other_args.append("+hydra.hydra_logging.root.handles=[]")
        other_args.append("hydra.hydra_logging.disable_existing_loggers=true")
        other_args.append("hydra.output_subdir=null")
    for oarg in other_args:
        sys.argv.append(oarg)
    sys.argv.append(f"+mode={mode}")
    main = hydra.main(config_path="conf", config_name=config)(run)
    main()

if __name__ == "__main__":
    moai()