from moai.utils.arguments import (
    ensure_path,
    ensure_choices,
)

import pickle
import torch
import typing
import logging
import os
import toolz

__all__ = ["Pkl"]

log = logging.getLogger(__name__)

class Pkl(typing.Callable[[typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]], None]):

    __MODES__ = ['all', 'append']
    
    def __init__(self,
        path:           str,
        keys:           typing.Union[str, typing.Sequence[str]],
        mode:           str="append", # all"
        counter_format: str="05d",
    ):
        self.mode = ensure_choices(log, "saving mode", mode, Pkl.__MODES__)
        self.folder = ensure_path(log, "output folder", path)
        self.index = 0
        self.keys = [keys] if type(keys) is str else list(keys)
        self.fmt = counter_format
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        arrays = { }
        for key in self.keys:
            split = key.split('.')
            arrays[key] = toolz.get_in(split, tensors).detach().cpu().numpy()
        mode = 'ab' if self.mode == 'append' else 'b'
        batch = toolz.get_in(['__moai__', 'batch_index'], tensors)
        step = toolz.get_in(['__moai__', 'optimization_step'], tensors)
        stage = toolz.get_in(['__moai__', 'optimization_stage'], tensors)
        save = { 
            'optimization_state': {
                'iteration': str(step),
                'stage': stage,
            }, 'parameters_state': arrays
        } if step else arrays
        with open(os.path.join(self.folder, f"results_{batch:{self.fmt}}.pkl"), mode) as f:
            pickle.dump(save, f)