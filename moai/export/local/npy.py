from moai.utils.arguments import (
    ensure_path,
    ensure_choices,
)

import numpy
import torch
import typing
import logging
import os
import toolz

__all__ = ["Npy"]

log = logging.getLogger(__name__)

class Npy(typing.Callable[[typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]], None]):

    __MODES__ = ['all', 'append']
    
    def __init__(self,
        path:           str,
        keys:           typing.Union[str, typing.Sequence[str]],
        mode:           str="combined", # all"
        counter_format: str="05d",
        combined:       bool=False,
        compressed:     bool=False,
    ):
        self.mode = ensure_choices(log, "saving mode", mode, Npy.__MODES__)
        self.folder = ensure_path(log, "output folder", path)
        self.index = 0
        self.keys = [keys] if type(keys) is str else list(keys)
        self.fmt = counter_format
        self.dict_mode = combined
        self.compressed = compressed
        self.ext = 'npz' if compressed else 'npy'
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        arrays = { }
        for key in self.keys:
            split = key.split('.')
            arrays[key] = toolz.get_in(split, tensors).detach().cpu().numpy()
        if self.mode == 'all':
            mode = 'ab'
            if self.dict_mode:
                with open(os.path.join(self.folder, f"{self.index:{self.fmt}}.{self.ext}"), mode) as f:
                    numpy.savez(f, arrays) if self.compressed else numpy.save(f, arrays)
            else:
                for key in self.keys:
                    with open(os.path.join(self.folder, f"{self.index:{self.fmt}}_{key}.{self.ext}"), mode) as f:
                        numpy.savez(f, arrays[key]) if self.compressed else numpy.save(f, arrays[key])
            self.index += 1
        else:
            log.error('Npy/Npz exporting is not yet enabled in append mode.')
        