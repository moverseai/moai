from moai.parameters.optimization.optimizers.larc import LARC as Outer

import torch
import omegaconf.omegaconf
import hydra.utils as hyu
import typing

__all__ = ['LARC']

#NOTE: modified from https://github.com/alphadl/lookahead.pytorch

class LARC(object):
    """ Implements the LARC algorithm.

    - **Paper**: [Large Batch Training of Convolutional Networks](https://arxiv.org/pdf/1708.03888.pdf)
    - **Implementation**: [GitHub @ NVIDIA](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py)

    """
    def __init__(self,
        parameters: typing.Iterator[torch.nn.Parameter],
        optimizers: omegaconf.DictConfig,
        trust_coefficient: float=0.02,
        clip: bool=True,
        eps: float=1e-8,
    ):
        self.optimizers = [
            Outer(
                optimizer=hyu.instantiate(optimizers[0], parameters), 
                trust_coefficient=trust_coefficient, clip=clip, eps=eps,
            )
        ]
