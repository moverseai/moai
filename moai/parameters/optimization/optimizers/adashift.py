import collections
import typing

import torch

__all__ = ["AdaShift"]

# NOTE: modified from https://github.com/MichaelKonobeev/adashift


class AdaShift(torch.optim.Optimizer):
    r"""Implements AdaShift algorithm.

    - **Paper**: [ADASHIFT: DECORRELATION AND CONVERGENCE OF ADAPTIVE LEARNING RATE METHODS](https://arxiv.org/pdf/1810.00143.pdf)
    - **Implementation**: [GitHub @ MichaelKonobeev](https://github.com/MichaelKonobeev/adashift)

    See https://openreview.net/forum?id=HkgTkhRcKQ

    Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate
      keep_num (int): number of gradients used to compute first moment
        estimation.
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square
      eps (float, optional): term added to the denominator to improve
        numerical stability
      reduce_func (callable or None): function applied to squared gradients
        to further reduce the correlation. If None, no function is applied.
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        keep_num: int = 10,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-10,
        reduce_func: typing.Callable = torch.max,
    ):
        beta1, _ = betas
        exp_weight_sum = sum(beta1**i for i in range(keep_num))
        first_grad_weight = beta1 ** (keep_num - 1) / exp_weight_sum
        last_grad_weight = 1.0 / exp_weight_sum
        defaults = dict(
            lr=lr,
            keep_num=keep_num,
            betas=betas,
            eps=eps,
            reduce_func=reduce_func,
            first_grad_weight=first_grad_weight,
            last_grad_weight=last_grad_weight,
        )
        super(AdaShift, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaShift does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 1
                    state["grad_deque"] = collections.deque(
                        [grad.clone()], maxlen=group["keep_num"]
                    )
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    continue

                grad_deque = state["grad_deque"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                grad_apply = len(grad_deque) == group["keep_num"]
                offset_grad = grad_deque[0]
                grad_deque.append(grad.clone())
                if not grad_apply:
                    continue

                first_grad_weight = group["first_grad_weight"]
                last_grad_weight = group["last_grad_weight"]
                beta1, beta2 = group["betas"]
                (
                    exp_avg.sub_(first_grad_weight, offset_grad)
                    .mul_(beta1)
                    .add_(last_grad_weight, grad)
                )

                reduce_func = group["reduce_func"] or (lambda x: x)
                reduced_grad_sq = reduce_func(offset_grad.mul_(offset_grad))
                exp_avg_sq.mul_(beta2).add_(1 - beta2, reduced_grad_sq)
                bias_correction = 1 - beta2 ** (state["step"] - group["keep_num"])
                denom = exp_avg_sq.div(bias_correction).sqrt_().add_(group["eps"])

                p.data.addcdiv_(-group["lr"], exp_avg, denom)
        return loss
