from moai.utils.arguments import (
    ensure_path,
    ensure_choices,
)

import numpy as np
import torch
import typing
import logging
import os
import toolz

__all__ = ["Npz"]

log = logging.getLogger(__name__)

class Npz(typing.Callable[[typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]], None]):

    __MODES__ = ['all', 'append']
    
    def __init__(self,
        path:           str,
        keys:           typing.Union[str, typing.Sequence[str]],
        mode:           str="all", # combined
        counter_format: str="05d",
        combined:       bool=False,
        compressed:     bool=False,
    ):
        self.mode = ensure_choices(log, "saving mode", mode, Npz.__MODES__)
        self.folder = ensure_path(log, "output folder", path)
        self.index = 0
        self.keys = [keys] if type(keys) is str else list(keys)
        self.fmt = counter_format
        self.dict_mode = combined
        self.compressed = compressed
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        arrays = { }
        for key in self.keys:
            split = key.split('.')
            arrays[key] = toolz.get_in(split, tensors).squeeze().detach().cpu().numpy()
        if self.mode == 'all':
            if self.dict_mode:
                f = os.path.join(self.folder, f"{self.index:{self.fmt}}.npz")
                np.savez_compressed(f, **arrays) if self.compressed else np.savez(f, **arrays)
            else:
                for key in self.keys:
                    f = os.path.join(self.folder, f"{self.index:{self.fmt}}_{key}.npz")
                    np.savez_compressed(f, arrays[key]) if self.compressed else np.savez(f, arrays[key])
            self.index += 1
        else:
            log.error('Npz exporting is not yet enabled in append mode.')
        