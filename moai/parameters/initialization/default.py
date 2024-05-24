import torch

import logging

log = logging.getLogger(__name__)

__all__ = ["Default"]

class Default(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, model: torch.nn.Module) -> None:
        log.info(f"Initializing model with default parameters")
        pass
        