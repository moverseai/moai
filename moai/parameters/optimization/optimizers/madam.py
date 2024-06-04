import typing

import torch


class IntegerMadam(torch.optim.Optimizer):
    r"""Implements the Integer Madam optimisation algorithm.

    ??? info "[Learning compositional functions via multiplicative weight updates](https://arxiv.org/pdf/2006.14560.pdf)"
        **Abstract**: Compositionality is a basic structural feature of both biological and artificial
        neural networks. Learning compositional functions via gradient descent incurs
        well known problems like vanishing and exploding gradients, making careful
        learning rate tuning essential for real-world applications. This paper proves that
        multiplicative weight updates satisfy a descent lemma tailored to compositional
        functions. Based on this lemma, we derive Madam—a multiplicative version of
        the Adam optimiser—and show that it can train state of the art neural network
        architectures without learning rate tuning. We further show that Madam is easily
        adapted to train natively compressed neural networks by representing their weights
        in a logarithmic number system. We conclude by drawing connections between
        multiplicative weight updates and recent findings about synapses in biology

    ??? info "[GitHub @ jxbz/madam](https://github.com/jxbz/madam)"

    Arguments:
        base_lr(float, optional):
        lr_factor(float, optional):
        levels(int, optional):
        p_scale(float, optional):
        g_bound(float, optional):
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        base_lr: float = 0.001,
        lr_factor: float = 10,
        levels: int = 4096,
        p_scale: float = 3.0,
        g_bound: float = 10.0,
    ):
        self.p_scale = p_scale
        self.g_bound = g_bound
        defaults = dict(base_lr=base_lr, lr_factor=lr_factor, levels=levels)
        super(IntegerMadam, self).__init__(params, defaults)

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

                lr = group["base_lr"]
                lr_factor = group["lr_factor"]
                levels = group["levels"]

                state = self.state[p]
                if len(state) == 0:
                    state["max"] = self.p_scale * (p * p).mean().sqrt().item()
                    state["integer"] = torch.randint_like(p, low=0, high=levels - 1)
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                bias_correction = 1 - 0.999 ** state["step"]
                state["exp_avg_sq"] = (
                    0.999 * state["exp_avg_sq"] + 0.001 * p.grad.data**2
                )

                g_normed = p.grad.data / (state["exp_avg_sq"] / bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-self.g_bound, self.g_bound)

                rounded = torch.round(g_normed * lr_factor) * torch.sign(p.data)
                state["integer"] = (state["integer"] + rounded).clamp_(0, levels - 1)
                p.data = (
                    torch.sign(p.data)
                    * state["max"]
                    * torch.exp(-lr * state["integer"])
                )

        return loss


class Madam(torch.optim.Optimizer):
    r"""Implements the Madam optimisation algorithm.

    ??? info "[Learning compositional functions via multiplicative weight updates](https://arxiv.org/pdf/2006.14560.pdf)"
        **Abstract**: Compositionality is a basic structural feature of both biological and artificial
        neural networks. Learning compositional functions via gradient descent incurs
        well known problems like vanishing and exploding gradients, making careful
        learning rate tuning essential for real-world applications. This paper proves that
        multiplicative weight updates satisfy a descent lemma tailored to compositional
        functions. Based on this lemma, we derive Madam—a multiplicative version of
        the Adam optimiser—and show that it can train state of the art neural network
        architectures without learning rate tuning. We further show that Madam is easily
        adapted to train natively compressed neural networks by representing their weights
        in a logarithmic number system. We conclude by drawing connections between
        multiplicative weight updates and recent findings about synapses in biology

    ??? info "[GitHub @ jxbz/madam](https://github.com/jxbz/madam)"

    Arguments:
        lr (float, required): The learning rate. 0.01 is a good initial value to try.
        p_scale(float, optional):
        g_bound(float, optional):
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 0.01,
        p_scale: float = 3.0,
        g_bound: float = 10.0,
    ):
        self.p_scale = p_scale
        self.g_bound = g_bound
        defaults = dict(lr=lr)
        super(Madam, self).__init__(params, defaults)

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
                if len(state) == 0:
                    state["max"] = self.p_scale * (p * p).mean().sqrt().item()
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                bias_correction = 1 - 0.999 ** state["step"]
                state["exp_avg_sq"] = (
                    0.999 * state["exp_avg_sq"] + 0.001 * p.grad.data**2
                )

                g_normed = p.grad.data / (state["exp_avg_sq"] / bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-self.g_bound, self.g_bound)

                p.data *= torch.exp(-group["lr"] * g_normed * torch.sign(p.data))
                p.data.clamp_(-state["max"], state["max"])

        return loss
