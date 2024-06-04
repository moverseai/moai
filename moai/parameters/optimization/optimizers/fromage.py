"""
Copyright (C) 2020 Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import math
import typing

import torch

# NOTE: from https://github.com/jxbz/fromage


class Fromage(torch.optim.Optimizer):
    r"""Implements the Fromage optimisation algorithm.

    ??? info "[On the distance between two neural networks and the stability of learning](https://arxiv.org/pdf/2002.03432.pdf)"
        **Abstract**: This paper relates parameter distance to gradient breakdown for a broad class of
        nonlinear compositional functions. The analysis leads to a new distance function
        called deep relative trust and a descent lemma for neural networks. Since the resulting learning rule seems not to require learning rate grid search, it may unlock a simpler workflow for training deeper and more complex neural networks. Please find
        the Python code used in this paper here: https://github.com/jxbz/fromage.

    ??? info "[GitHub @ jxbz/fromage](https://github.com/jxbz/fromage)"

    Arguments:
        lr (float, required): The learning rate. 0.01 is a good initial value to try.
        p_bound (float, optional): Restricts the optimisation to a bounded set. A
            value of 2.0 restricts parameter norms to lie within 2x their
            initial norms. This regularises the model class.
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 0.01,
        p_bound: float = None,
    ):
        self.p_bound = p_bound
        defaults = dict(lr=lr)
        super(Fromage, self).__init__(params, defaults)

    def step(
        self, closure: typing.Optional[typing.Callable[[], float]] = None
    ) -> typing.Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0 and self.p_bound is not None:
                    state["max"] = self.p_bound * p.norm().item()

                d_p = p.grad.data
                d_p_norm = p.grad.norm()
                p_norm = p.norm()

                if p_norm > 0.0 and d_p_norm > 0.0:
                    p.data.add_(-group["lr"], d_p * (p_norm / d_p_norm))
                else:
                    p.data.add_(-group["lr"], d_p)
                p.data /= math.sqrt(1 + group["lr"] ** 2)

                if self.p_bound is not None:
                    p_norm = p.norm().item()
                    if p_norm > state["max"]:
                        p.data *= state["max"] / p_norm

        return loss
