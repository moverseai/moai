import torch
import typing
import logging
import sys
import functools

__all__ = ["Empty"]

log = logging.getLogger(__name__)

class Empty(torch.utils.data.Dataset):
    def __init__(self):
        log.warning("Constructing an empty dataset !")
    
    def __len__(self) -> int:
        return sys.maxsize

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return {}

class Dummy(torch.utils.data.Dataset):
    def __init__(self,
        size:       int=sys.maxsize,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        self.generators = {}
        self.size = size        
        for key, props in kwargs.items():
            self.generators[key] = functools.partial(getattr(torch, props.init or 'rand'),
                size=tuple(props.shape),
            )
        log.info(f"Using a custom dataset producing {len(kwargs)} dummy tensors ([{list(kwargs.keys())}]) of size {len(self)}.")
    
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict((k, g()) for k, g in self.generators.items())