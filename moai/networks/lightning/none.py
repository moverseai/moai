import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import inspect

log = logging.getLogger(__name__)

__all__ = ['Null']

from moai.monads.execution.cascade import _create_accessor

class Null(minet.FeedForward):
    def __init__(self,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        feedforward:    omegaconf.DictConfig=None,
        monads:         omegaconf.DictConfig=None,
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(Null, self).__init__(
            feedforward=feedforward, monads=monads, 
            visualization=visualization, data=data,
            supervision=supervision, validation=validation, 
            export=export, parameters=parameters,
        )
    
    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        return td
        