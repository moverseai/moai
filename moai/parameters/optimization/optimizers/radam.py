import math
import typing

import torch

__all__ = ["RAdam"]

# NOTE: see this https://github.com/LiyuanLucasLiu/RAdam/pull/51 for 1.6 and upwards compatibility


class RAdam(torch.optim.Optimizer):
    r"""Implements the Rectified Adam (RAdam) optimization algorithm.

    ??? info "Paper"
        [![Paper](https://img.shields.io/static/v1?label=1711.06753&message=ON THE VARIANCE OF THE ADAPTIVE LEARNING RATE AND BEYOND&color=00bdd6&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1908.03265.pdf)

    ??? info "Repository"
        [![Repo](https://github-readme-stats.vercel.app/api/pin/?username=LiyuanLucasLiu&repo=RAdam)](https://github.com/LiyuanLucasLiu/RAdam)

    ???+ important "Config Entry"
        ```yaml
          - model/parameters/optimization: single # when used as a single optimizer
          - model/parameters/optimization/optimizer: radam
        ```

    ???+ important "Config Parameters (Default)"
        ```yaml
        radam:
            lr: 0.0001
            betas: [0.9, 0.999]
            eps: 1e-8
            weight_decay: 0.0
            degenerated_to_sgd: true
        ```

    Arguments:
        lr (float, required):
            Optimizer's learning rate.
            **Default value: 1e-3.**
        betas (float):
            The momentum values.
            **Default value: [0.9, 0.999].**
        eps (float):
            Small epsilon value for numerical stabilization/
            **Default value: 1e-8.**
        weight_decay (float):
            Weight decay parameter.
            **Default value: 0.0.**
        degenerated_to_sgd (bool):
            Allow for degeneration to SGD with momentum.
            **Default value: True.**
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        degenerated_to_sgd: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # NOTE: for older pytorch versions this was the addcmul API
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # NOTE: for recent pytorch version this was changed to:
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        # NOTE: same API issue as above
                        # p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    # NOTE: same API issue as above
                    # p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        # NOTE: same API issue as above
                        # p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                        p_data_fp32.add_(
                            p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                        )
                    # NOTE: same API issue as above
                    # p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(torch.optim.Optimizer):
    r"""Implements the Plain Rectified Adam (RAdam) optimization algorithm, without caching optimization.

    ??? info "Paper"
        [![Paper](https://img.shields.io/static/v1?label=1711.06753&message=ON THE VARIANCE OF THE ADAPTIVE LEARNING RATE AND BEYOND&color=00bdd6&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1908.03265.pdf)

    ??? info "Repository"
        [![Repo](https://github-readme-stats.vercel.app/api/pin/?username=LiyuanLucasLiu&repo=RAdam)](https://github.com/LiyuanLucasLiu/RAdam)

    ???+ important "Config Entry"
        ```yaml
          - model/parameters/optimization: single # when used as a single optimizer
          - model/parameters/optimization/optimizer: plain_radam
        ```

    ???+ important "Config Parameters (Default)"
        ```yaml
        radam:
            lr: 0.0001
            betas: [0.9, 0.999]
            eps: 1e-8
            weight_decay: 0.0
            degenerated_to_sgd: true
        ```

    Arguments:
        lr (float, required):
            Optimizer's learning rate.
            **Default value: 1e-3.**
        betas (float):
            The momentum values.
            **Default value: [0.9, 0.999].**
        eps (float):
            Small epsilon value for numerical stabilization/
            **Default value: 1e-8.**
        weight_decay (float):
            Weight decay parameter.
            **Default value: 0.0.**
        degenerated_to_sgd (bool):
            Allow for degeneration to SGD with momentum.
            **Default value: True.**
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        degenerated_to_sgd: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    step_size = (
                        group["lr"]
                        * math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                        / (1 - beta1 ** state["step"])
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    step_size = group["lr"] / (1 - beta1 ** state["step"])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(torch.optim.Optimizer):
    """ """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup: float = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["warmup"] > state["step"]:
                    scheduled_lr = 1e-8 + state["step"] * group["lr"] / group["warmup"]
                else:
                    scheduled_lr = group["lr"]

                step_size = (
                    scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                )

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
