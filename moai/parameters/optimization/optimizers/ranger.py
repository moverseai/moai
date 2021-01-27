from moai.parameters.optimization.optimizers.lookahead import Lookahead

import math
import torch
import typing

__all__ = ['Ranger']

#NOTE: from https://github.com/mgrankin/over9000/blob/master/ranger.py

class Ranger(Lookahead):
     """Implements a combination of RAdam and Lookahead:

     - [ON THE VARIANCE OF THE ADAPTIVE LEARNING RATE AND BEYOND](https://arxiv.org/pdf/1908.03265.pdf)
     - [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610.pdf)
     - **Implementation**: [GitHub @ mgrankin](https://github.com/mgrankin/over9000)

     """
     def __init__(self,
          params: typing.Iterator[torch.nn.Parameter],
          alpha: float=0.5,
          k: int=5,
          lr: float=1e-3,
          betas: typing.Tuple[float, float]=(0.9, 0.999),
          eps: float=1e-8,
          weight_decay: float=0,
          degenerated_to_sgd: bool=True,
     ):
          super(Ranger, self).__init__(
               mio.RAdam(params, 
                    lr=lr, betas=betas, eps=eps,
                    weight_decay=weight_decay,
                    degenerated_to_sgd=degenerated_to_sgd,
               ), k=k, alpha=alpha
          )