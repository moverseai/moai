import moai.parameters.optimization.optimizers as mio
from moai.parameters.optimization.optimizers.lookahead import Lookahead

import torch
import math
import torch
import typing

__all__ = ['RangerLars']

#NOTE: from https://github.com/mgrankin/over9000/blob/master/rangerlars.py

class RangerLars(Lookahead):
     """Implements a combination of RAdam, LARS and Lookahead
     
     - [ON THE VARIANCE OF THE ADAPTIVE LEARNING RATE AND BEYOND](https://arxiv.org/pdf/1908.03265.pdf)
     - [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610.pdf)
     - [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)
     - **Implementation**: [GitHub @ mgrankin](https://github.com/mgrankin/over9000)

     """
     def __init__(self,
          params: typing.Iterator[torch.nn.Parameter],
          alpha: float=0.5,
          k: int=5,
          lr: float=1e-3,
          betas: typing.Tuple[float, float]=(0.9, 0.999),
          eps: float=1e-8,
          weight_decay: float=0
     ):
          super(RangerLars, self).__init__(
               mio.Ralamb(params,
                    lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
               ), k=k, alpha=alpha
          )