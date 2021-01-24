import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import inspect

log = logging.getLogger(__name__)

__all__ = ['Custom']

class Custom(minet.FeedForward):
    def __init__(self,
        custom:         omegaconf.DictConfig,
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
        super(Custom, self).__init__(
            feedforward=feedforward, monads=monads, 
            visualization=visualization, data=data,
            supervision=supervision, validation=validation, 
            export=export, parameters=parameters,
        )         
        self.model = hyu.instantiate(custom)
        self.fwds = []
        params = inspect.signature(self.model.forward).parameters
        model_in = list(zip(*[mirtp.force_list(configuration.io[prop]) for prop in params]))
        model_out = mirtp.split_as(mirtp.resolve_io_config(configuration.io['out']), model_in)
        self.res_fill = [mirtp.get_result_fillers(self.model, out) for out in model_out]
        get_filler = iter(self.res_fill)
        for keys in model_in:
            self.fwds.append(lambda td,
                tk=keys,
                args=params.keys(),
                model=self.model,
                filler=next(get_filler):
                    filler(td, model(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            )
    
    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for f in self.fwds:
            f(td)
        return td
        