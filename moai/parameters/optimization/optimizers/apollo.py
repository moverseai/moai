import typing

import torch

__all__ = ["Apollo", "ApolloW"]

# NOTE: from https://github.com/XuezheMax/apollo/blob/master/optim/apollo.py


class Apollo(torch.optim.Optimizer):
    r"""Implements Atom algorithm.

    - **Paper**: [Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization](https://arxiv.org/pdf/2009.13586.pdf)
    - **Implementation**: [GitHub @ XuezheMax](https://github.com/XuezheMax/apollo)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        beta (float, optional): coefficient used for computing
            running averages of gradient (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        warmup (int, optional): number of warmup steps (default: 0)
        init_lr (float, optional): initial learning rate for warmup (default: 0.01)
        weight_decay (float, optional): weight decay coefficient (default: 0)

    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        # lr: float=torch.optim.optimizer.required,
        lr=None,
        beta: float = 0.9,
        eps: float = 1e-4,
        warmup: int = 100,
        init_lr: float = 0.01,
        weight_decay: float = 0,
    ):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if not 0.0 <= init_lr <= 1.0:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
        )
        super(Apollo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Apollo, self).__setstate__(state)

    @torch.no_grad()
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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["approx_hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Previous update direction
                    state["update"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # Calculate current lr
                if state["step"] < group["warmup"]:
                    curr_lr = (group["base_lr"] - group["init_lr"]) * state[
                        "step"
                    ] / group["warmup"] + group["init_lr"]
                else:
                    curr_lr = group["lr"]

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Atom does not support sparse gradients.")

                # Perform step weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                beta = group["beta"]
                exp_avg_grad = state["exp_avg_grad"]
                B = state["approx_hessian"]
                d_p = state["update"]

                state["step"] += 1
                bias_correction = 1 - beta ** state["step"]
                alpha = (1 - beta) / bias_correction

                # Update the running average grad
                delta_grad = grad - exp_avg_grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(group["eps"])
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = (
                    delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha)
                    - B.mul(v_sq).sum()
                )

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                denom = B.abs().clamp_(min=1)
                d_p.copy_(exp_avg_grad.div(denom))

                p.add_(d_p, alpha=-curr_lr)

        return loss


class ApolloW(torch.optim.Optimizer):
    r"""Implements Atom algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        beta (float, optional): coefficient used for computing
            running averages of gradient (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        warmup (int, optional): number of warmup steps (default: 0)
        init_lr (float, optional): initial learning rate for warmup (default: 0.01)
        weight_decay (float, optional): weight decay coefficient (default: 0)

    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        # lr: float=torch.optim.optimizer.required,
        lr=None,
        beta: float = 0.9,
        eps: float = 1e-4,
        warmup: int = 100,
        init_lr: float = 0.01,
        weight_decay: float = 0,
    ):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if not 0.0 <= init_lr <= 1.0:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
        )
        super(ApolloW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ApolloW, self).__setstate__(state)

    @torch.no_grad()
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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["approx_hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Previous update direction
                    state["update"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # Calculate current lr
                if state["step"] < group["warmup"]:
                    curr_lr = (group["base_lr"] - group["init_lr"]) * state[
                        "step"
                    ] / group["warmup"] + group["init_lr"]
                else:
                    curr_lr = group["lr"]

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Atom does not support sparse gradients.")

                beta = group["beta"]
                exp_avg_grad = state["exp_avg_grad"]
                B = state["approx_hessian"]
                d_p = state["update"]

                state["step"] += 1
                bias_correction = 1 - beta ** state["step"]
                alpha = (1 - beta) / bias_correction

                # Update the running average grad
                delta_grad = grad - exp_avg_grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(group["eps"])
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = (
                    delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha)
                    - B.mul(v_sq).sum()
                )

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                denom = B.abs().clamp_(min=1)
                d_p.copy_(exp_avg_grad.div(denom))

                # Perform step weight decay
                if group["weight_decay"] != 0:
                    d_p.add_(p, alpha=group["weight_decay"])

                p.add_(d_p, alpha=-curr_lr)

        return loss
