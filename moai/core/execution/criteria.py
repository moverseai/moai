from moai.core.execution.constants import Constants as C

import moai.core.execution.common as mic
import typing
import torch
import toolz
import omegaconf.omegaconf
import hydra.utils as hyu
import inspect

__all__ = ['Criteria']

class Criteria():    
    def __init__(self, 
        criteria: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ) -> None:
        self.operations = []
        for k in kwargs or {}:
            params = kwargs[k] #NOTE: list args for multi calling
            override_params = params.get(C._PARAMS_, None) or {}            
            target = criteria[k]
            operation = hyu.instantiate(target, **override_params)
            signature = inspect.signature(operation)
            extras = mic.__PRESET_ARGS__.intersection(signature.parameters.keys())            
            #NOTE: operate should change to partial func call
            if mic._is_first_arg_tensor_dict(next(iter(signature.parameters.values()))):# signature.parameters[next(iter(args))]):
                args = set(toolz.drop(1, signature.parameters.keys())) - mic.__PRESET_ARGS__
                op_args = toolz.keyfilter(lambda k: k in args, params) #NOTE: currently assumes that no arg is named `params`
                self.operations.append(mic.CriteriaDictOperation(operation, op_args, extras))
            else:
                args = signature.parameters.keys() - mic.__PRESET_ARGS__
                op_args = toolz.keyfilter(lambda k: k in args, params) #NOTE: currently assumes that no arg is named `params`
                tensor_args = mic._get_tensor_args(signature.parameters)
                self.operations.append(mic.CriteriaArgsOperation(operation, 
                    toolz.dissoc(op_args,*(args-tensor_args.keys())), 
                    toolz.dissoc(op_args,*tensor_args.keys()),
                    extras))

    def __call__(self, 
        tensors:    typing.Mapping[str, torch.Tensor], 
        extras:     typing.Mapping[str, typing.Any],
    ) -> None:
        stop = False
        for operation in self.operations:
            stop = operation(tensors[C._MOAI_METRICS_], extras)
            if stop:
                break
        return stop