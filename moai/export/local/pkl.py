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

__all__ = ["Pkl", "append_pkl"]

log = logging.getLogger(__name__)

from abc import ABC, abstractmethod

class TensorMonitor(ABC): #NOTE: use as base to enforce some args
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is TensorMonitor:
            subclass_dict = subclass.__mro__[0].__dict__
            cls_dict = cls.__mro__[0].__dict__
            cls_abstract_methods = cls.__abstractmethods__
            
            for method_name, method in cls_dict.items():
                # If the method is an abstracmethod
                if method_name in cls_abstract_methods and hasattr(method,'__annotations__'):
                    if method_name not in subclass_dict or \
                    subclass_dict[method_name].__annotations__ != cls_dict[method_name].__annotations__:
                            return False
            return True
    
    @abstractmethod
    def a(self, str_param: str) -> list:
        raise NotImplementedError
    
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
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
        batch_idx:  typing.Optional[int]=None,
        optimization_step: typing.Optional[int]=None,
        stage: typing.Optional[str]=None,
    ) -> None:
        arrays = { }
        for key in self.keys:
            split = key.split('.')
            arrays[key] = toolz.get_in(split, tensors).detach().cpu().numpy()
        if self.mode == 'append':
            mode = 'ab'
            # batch = toolz.get_in(['__moai__', 'batch_index'], tensors)
            # step = toolz.get_in(['__moai__', 'optimization_step'], tensors)
            # stage = toolz.get_in(['__moai__', 'optimization_stage'], tensors)            
            save = { 
                'optimization_state': {
                    'iteration': str(optimization_step),
                    'stage': stage,
                }, 'parameters_state': arrays
            } if step is not None else arrays
            with open(os.path.join(self.folder, f"results_{batch_idx:{self.fmt}}.pkl"), mode) as f:
                pickle.dump(save, f)
        else:
            mode = 'b'
            log.error("Pickle exporting is not yet enabled in non append mode.")
        
def append_pkl(
        tensors:            typing.Dict[str, torch.Tensor],
        path:               str,
        keys:               typing.Union[str, typing.Sequence[str]],
        lightning_step:     typing.Optional[int]=None,
        batch_idx:          typing.Optional[int]=None,
        optimization_step:  typing.Optional[int]=None,
        stage:              typing.Optional[str]=None,
        fmt:                str="05d",
) -> None:
    arrays = { }
    for key in keys:
        split = key.split('.')
        arrays[key] = toolz.get_in(split, tensors).detach().cpu().numpy()
    save = { 
        'optimization_state': {
            'iteration': str(optimization_step),
            'stage': stage,
        }, 'parameters_state': arrays
    } if step is not None else arrays
    with open(os.path.join(path, f"new_results_{batch_idx:{fmt}}.pkl"), 'ab') as f:
        pickle.dump(save, f)

class _Pkl(typing.Callable[[typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]], None]):

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
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        arrays = { }
        for key in self.keys:
            split = key.split('.')
            arrays[key] = toolz.get_in(split, tensors).detach().cpu().numpy()
        if self.mode == 'append':
            mode = 'ab'
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
        else:
            mode = 'b'
            log.error("Pickle exporting is not yet enabled in non append mode.")
        