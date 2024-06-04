try:
    # from moai.action.diff import diff
    # from moai.action.plot import plot
    from moai.action.archive import archive
    from moai.action.export import export
    from moai.action.run import run
except:
    # from action.diff import diff
    # from action.plot import plot
    from action.archive import archive
    from action.export import export
    from action.run import run

import logging
import os
import sys

import hydra

logging.captureWarnings(True)

from rich.traceback import install as traceback_install

traceback_install(
    width=120,
    extra_lines=5,
    theme=None,
    word_wrap=True,
    show_locals=False,
    indent_guides=True,
)

ERROR_FORMAT = "%(levelname)s at %(asctime)s in %(funcName)s in %(filename) at line %(lineno)d: %(message)s"
DEBUG_FORMAT = "%(lineno)d in %(filename)s at %(asctime)s: %(message)s"
FORMAT = "[%(levelname)s] - %(message)s"

logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger("moai")


def debug(cfg):
    log.info(cfg)


__MODES__ = {
    # 'diff': diff,
    # 'plot': plot,
    "debug": debug,
    "archive": archive,
    "export": export,
    "run": run,
}

__MIN_ARGS_COUNT__ = {
    # 'diff': 1,
    # 'plot': 1,
    "debug": 2,
    "archive": 2,
    "export": 2,
    "run": 2,
}


def moai():
    mode = sys.argv.pop(1)
    action = sys.argv.pop(1) if mode != "archive" else mode
    if mode not in __MODES__:
        log.error(
            f"Wrong moai execution mode ({mode}), supported modes are: {list(__MODES__.keys())}."
        )
        sys.exit(-1)
    min_args = __MIN_ARGS_COUNT__[mode]
    if len(sys.argv) < min_args:
        log.error(
            f"Insufficient arguments provided for moai. Calling should specify the mode and config file: 'moai MODE CONFIG'."
        )
    config = sys.argv.pop(1) if min_args > 1 else f"tools/{mode}.yaml"
    other_args = ["hydra.job.chdir=True"]
    if min_args > 1 or mode == "plot":
        output_dir = (
            "hydra.run.dir=actions/"
            + action
            + "/${now:%Y-%m-%d}/${now:%H-%M-%S}-${oc.select:experiment.name,moai-run}"
        )
        other_args.append(output_dir)
    elif mode == "resume":
        output_dir = f"hydra.run.dir={sys.argv.pop(1)}"
        other_args.append(output_dir)
    else:
        other_args.append("+hydra.hydra_logging.root.handles=[]")
        other_args.append("hydra.hydra_logging.disable_existing_loggers=true")
        other_args.append("hydra.output_subdir=null")
    for oarg in other_args:
        sys.argv.append(oarg)
    if mode != "reprod":
        sys.argv.append(f"+mode={mode}")
        # add action to config
        sys.argv.append(f"+action={action}")
    else:
        sys.argv.append(f"+reprod={mode}")
    file_name = os.path.splitext(os.path.basename(config))[0]
    base_path = os.path.dirname(config)
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.getcwd(), base_path)
    main = hydra.main(config_path=base_path, config_name=file_name, version_base="1.3")(
        __MODES__[mode]
    )
    main()


if __name__ == "__main__":
    moai()
