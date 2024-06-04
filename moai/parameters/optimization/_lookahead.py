import typing

import hydra.utils as hyu
import omegaconf.omegaconf
import torch

from moai.parameters.optimization.optimizers.lookahead import Lookahead as Outer

__all__ = ["Lookahead"]

# NOTE: modified from https://github.com/alphadl/lookahead.pytorch


class Lookahead(object):
    """Implements the Lookahead algorithm.

    - **Paper**: [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610.pdf)
    - **Implementation**: [GitHub @ alphadl](https://github.com/alphadl/lookahead.pytorch)

    """

    def __init__(
        self,
        parameters: typing.Iterator[torch.nn.Parameter],
        optimizers: omegaconf.DictConfig,
        k: int = 5,
        alpha: float = 0.5,
    ):
        self.optimizers = [
            Outer(
                optimizer=hyu.instantiate(optimizers[0], parameters), k=k, alpha=alpha
            )
        ]
