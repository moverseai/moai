import torch

import logging

log = logging.getLogger(__name__)

__all__ = ["Default"]

class Default(object):
    def __init__(self):
        pass

    def __call__(self, model: torch.nn.Module) -> None:
        pass