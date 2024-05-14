try:
    from moai.action.train import train
    from moai.action.evaluate import evaluate
    from moai.action.play import play
    from moai.action.diff import diff
    from moai.action.plot import plot
    from moai.action.reprod import reprod
    from moai.action.fit import fit
    from moai.action.archive import archive
    from moai.action.export import export
    from moai.action.resume import resume
    from moai.action.run import run
except:
    from action.train import train
    from action.evaluate import evaluate
    from action.play import play
    from action.diff import diff
    from action.plot import plot
    from action.reprod import reprod
    from action.fit import fit
    from action.archive import archive
    from action.export import export
    from action.resume import resume
    from action.run import run

import omegaconf.omegaconf
import hydra
import logging
import sys
import os

logging.captureWarnings(True)

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
    'reprod': reprod,
    'fit': fit,
    'archive': archive,
    'export': export,
    'resume': resume,
    'run': run,
}

__MIN_ARGS_COUNT__ = {
    'train': 2,
    'evaluate': 2,
    'play': 2,
    'diff': 1,
    'plot': 1,
    'debug': 2,
    'reprod': 2,
    'fit': 2,
    'archive': 2,
    'export': 2,
    'resume': 1,
    'run': 2,
}

def run(cfg: omegaconf.DictConfig) -> None:
    reprod_key = "reprod"
    if not reprod_key in cfg:
        __MODES__[cfg.mode](cfg)
    else:
        __MODES__[cfg.reprod](cfg)

def moai():
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    mode = sys.argv.pop(1)
    action = sys.argv.pop(1) if mode != 'archive' else mode
    if mode not in __MODES__:
        log.error(f"Wrong moai execution mode ({mode}), supported modes are: {list(__MODES__.keys())}.")
        sys.exit(-1)
    min_args = __MIN_ARGS_COUNT__[mode]
    if len(sys.argv) < min_args:
        log.error(f"Insufficient arguments provided for moai. Calling should specify the mode and config file: \'moai MODE CONFIG\'.")    
    config = sys.argv.pop(1) if min_args > 1 else f"tools/{mode}.yaml" 
    other_args = ["hydra.job.chdir=True"]
    if min_args > 1 or mode == 'plot':
        output_dir = "hydra.run.dir=actions/" + action + "/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}" 
        other_args.append(output_dir)
    elif mode == 'resume':
        output_dir =  f"hydra.run.dir={sys.argv.pop(1)}"
        other_args.append(output_dir)
    else:
        other_args.append("+hydra.hydra_logging.root.handles=[]")
        other_args.append("hydra.hydra_logging.disable_existing_loggers=true")
        other_args.append("hydra.output_subdir=null")
    for oarg in other_args:
        sys.argv.append(oarg)
    if mode != 'reprod':
        sys.argv.append(f"+mode={mode}")
        # add action to config
        sys.argv.append(f"+action={action}")
    else:
        sys.argv.append(f"+reprod={mode}")
    file_name = os.path.splitext(os.path.basename(config))[0]
    base_path = os.path.dirname(config)
    # main = hydra.main(config_path="conf", config_name=config)(run)
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.getcwd(), base_path)
    main = hydra.main(config_path=base_path, config_name=file_name, version_base='1.3')(run)
    # main = hydra.main(config_path=None, config_name=config)(run)
    main()

if __name__ == "__main__":
    moai()