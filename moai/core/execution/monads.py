import moai.core.execution.common as mic
import moai.core.execution.expression as mie
import typing
import torch
import omegaconf.omegaconf
import hydra.utils as hyu
import inspect
import toolz
import logging
import lark

log = logging.getLogger(__name__)

__all__ = ['Monads']

class Monads(torch.nn.ModuleDict):
    execs: typing.List[typing.Callable]

    def __init__(self, 
        monads: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        errors = [k for k in kwargs if k not in monads]
        if errors:
            log.error(f"The following monads [{errors}] were not found in the configuration and will be ignored!")
        self.execs = []
        for i, key in enumerate(kwargs):
            if key not in monads:
                log.warning(f"Skipping monad '{key}' as it is not found in the configuration. This may lead to downstream errors.")
                continue
            try:
                self.add_module(key, hyu.instantiate(monads[key])) #TODO: stateless monads can be re-used
            except Exception as e:
                log.error(f"Could not instantiate the monad '{key}': {e}")
                continue
            graph_params = kwargs[key]
            sig = inspect.signature(self[key].forward)
            sig_params = list(filter(lambda p: p in graph_params, sig.parameters))
            extra_params = set(graph_params.keys()) - set(sig_params) - set(['out'])
            if extra_params:
                log.error(f"The parameters [{extra_params}] are not part of the `{key}` monad signature.")
            graph_params = mic._dict_of_lists_to_list_of_dicts(graph_params)
            for j, params in enumerate(graph_params):
                for k, v in params.items():
                    if isinstance(v, lark.Tree):
                        tmp_key = f"{key}{i}/{k}{j}"
                        self.add_module(tmp_key, mie.TreeModule(tmp_key, v))
                        self.execs.append((tmp_key, tmp_key, None))
                        params[k] = tmp_key
                self.execs.append((key, params['out'], toolz.dissoc(params, 'out')))

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for key, out, kwargs in self.execs:
            if kwargs:
                tensors[out] = self[key](**toolz.valmap(lambda v: tensors[v], kwargs))
            else:
                tensors[out] = self[key](tensors)
                # self[key](tensors)
        return tensors