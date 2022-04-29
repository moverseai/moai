import sys
import hydra
import os
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['Import']

class Import(object):
    def __init__(self,
        run_path:               bool=True,
        other_paths:            typing.Sequence[str]=[],
    ):
        if run_path:            
            cwd = hydra.utils.get_original_cwd() \
                if hydra.utils.HydraConfig.initialized() else os.getcwd()
            sys.path.append(cwd)
            log.info(f"Adding current working directory [{os.path.abspath(cwd)}] to path.")
        for path in other_paths:
            sys.path.append(path)
            log.info(f"Adding [{os.path.abspath(path)}] to path.")