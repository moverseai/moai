from moai.parameters.optimization.optimizers.swa import SWA as Outer

import torch
import omegaconf.omegaconf
import hydra.utils as hyu
import typing

__all__ = ['SWA']

#NOTE: modified from https://github.com/alphadl/lookahead.pytorch

class SWA(object):
    """Implements Stochastic Weight Averaging (SWA).

        - **Paper**: [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/pdf/1803.05407.pdf)
        - **Implementation**: [GitHub @ pytorch](https://github.com/pytorch/contrib/tree/master/torchcontrib)

    """
    def __init__(self,
        parameters: typing.Iterator[torch.nn.Parameter],
        optimizers: omegaconf.DictConfig,        
        swa_start=None,
        swa_freq=None,
        swa_lr=None
    ):
        self.optimizers = [
            Outer(
                optimizer=hyu.instantiate(optimizers[0], parameters), 
                swa_start=swa_start, swa_lr=swa_lr, swa_freq=swa_freq,
            )
        ]
