from moai.utils.arguments import assert_choices

import torch
import logging

log = logging.getLogger(__name__)

class Passthrough(torch.nn.Module):

    __MODES__ = ['minimize', 'maximize']

    def __init__(self, 
        mode:               str='minimize', # one of ['minimize', 'maximize']
    ):
        super(Passthrough, self).__init__()
        assert_choices(log, "mode", mode, Passthrough.__MODES__)
        self.fwd_func = lambda t: t if mode == 'minimize'\
            else lambda t: -t

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        return self.fwd_func(error)
