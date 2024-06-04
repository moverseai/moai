import typing

import torch

# NOTE: from https://github.com/jxbz/madam


class SignSGD(torch.optim.Optimizer):
    r"""Implements the Sign SGD optimisation algorithm.

    ??? info "[signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/pdf/1802.04434.pdf)"
        **Abstract**: Training large neural networks requires distributing learning across multiple workers, where the
        cost of communicating gradients can be a significant bottleneck. SIGNSGD alleviates this problem by transmitting just the sign of each minibatch
        stochastic gradient. We prove that it can get the best of both worlds: compressed gradients and
        SGD-level convergence rate. The relative `1/`2 geometry of gradients, noise and curvature informs whether SIGNSGD or SGD is theoretically
        better suited to a particular problem. On the practical side we find that the momentum counterpart
        of SIGNSGD is able to match the accuracy and convergence speed of ADAM on deep Imagenet
        models. We extend our theory to the distributed setting, where the parameter server uses majority
        vote to aggregate gradient signs from each worker enabling 1-bit compression of worker-server communication in both directions. Using a theorem
        by Gauss (1823) we prove that majority vote can achieve the same reduction in variance as full
        precision distributed SGD. Thus, there is great promise for sign-based optimisation schemes to
        achieve fast communication and fast convergence. Code to reproduce experiments is to be found at
        https://github.com/jxbz/signSGD.

    ??? info "[GitHub @ jxbz/madam](https://github.com/jxbz/madam)"

    Arguments:
        lr (float, required): The learning rate.
    """

    def __init__(self, params: typing.Iterator[torch.nn.Parameter], lr: float):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SignSGD, self).__init__(params, defaults)

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

                sign = torch.sign(p.grad.data)
                p.data.add_(-group["lr"], sign)

        return loss
