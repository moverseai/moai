import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import inspect

log = logging.getLogger(__name__)

__all__ = ['Wrapper']

from moai.monads.execution.cascade import _create_accessor

class Wrapper(minet.FeedForward):
    def __init__(self,
        inner:          omegaconf.DictConfig,
        configuration:  omegaconf.DictConfig,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        feedforward:    omegaconf.DictConfig=None,
        monads:         omegaconf.DictConfig=None,
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(Wrapper, self).__init__(
            feedforward=feedforward, monads=monads, 
            visualization=visualization, data=data,
            supervision=supervision, validation=validation, 
            export=export, parameters=parameters,
        )         
        self.model = hyu.instantiate(inner)
        self.fwds = []
        params = inspect.signature(self.model.forward).parameters
        model_in = list(zip(*[mirtp.force_list(configuration.io[prop]) for prop in params]))
        model_out = mirtp.split_as(mirtp.resolve_io_config(configuration.io['out']), model_in)
        self.res_fill = [mirtp.get_result_fillers(self.model, out) for out in model_out]
        get_filler = iter(self.res_fill)
        for keys in model_in:
            accessors = [_create_accessor(k if isinstance(k, str) else k[0]) for k in keys]  
            self.fwds.append(lambda td,
                tk=keys,
                acc=accessors,
                args=params.keys(),
                model=self.model,
                filler=next(get_filler):
                    filler(td, model(**dict(zip(args,
                        list(acc[i](td) if type(k) is str
                                else list(td[j] for j in k)
                            for i, k in enumerate(tk)
                        )
                    ))))
            )
    
    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for f in self.fwds:
            f(td)
        return td
        