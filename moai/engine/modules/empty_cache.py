import torch
import logging

log = logging.getLogger(__name__)

__all__ = ['EmptyCache']

class EmptyCache(object):
    def __init__(self, 

    ):
        log.info(f"Emptying PyTorch CUDA cache.")        
        torch.cuda.empty_cache()