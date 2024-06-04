import math
import typing

import torch


class SAdam(torch.optim.Optimizer):
    """Implements SAdam/SAmsgrad algorithm.

    - **Paper**: [Calibrating the Learning Rate for Adaptive Gradient Methods to Improve Generalization Performance](https://arxiv.org/pdf/1908.00700v1.pdf)
    - **Implementation**: [GitHub @ neilliang90/Sadam](https://github.com/neilliang90/Sadam/)

        Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, required): learning rate (default: 1e-1)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        partial (float, optional): partially adaptive parameter
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-1,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = True,
        partial: float = 1 / 4,
        transformer: str = "softplus",
        grad_transf: str = "square",
        smooth: float = 50,
        hist: bool = False,
    ):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            partial=partial,
            transformer=transformer,
            hist=hist,
            grad_transf=grad_transf,
            smooth=smooth,
        )
        super(SAdam, self).__init__(params, defaults)

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
        denom_list = []
        denom_inv_list = []
        m_v_eta = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]
                partial = group["partial"]
                grad_transf = group["grad_transf"]
                smooth = group["smooth"]

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                ################ tranformer for grad ###########################################################
                if grad_transf == "square":
                    grad_tmp = grad**2
                elif grad_transf == "abs":
                    grad_tmp = grad.abs()

                exp_avg_sq.mul_(beta2).add_((1 - beta2) * grad_tmp)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.clone()
                else:
                    denom = exp_avg_sq.clone()
                if grad_transf == "square":
                    denom.sqrt_()

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                ################ tranformer for denom ###########################################################
                if group["transformer"] == "softplus":
                    sp = torch.nn.Softplus(smooth)
                    denom = sp(denom)
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p.data.addcdiv_(
                        -step_size, exp_avg, (denom.add_(group["eps"])) ** (partial * 2)
                    )
                ################ logger ###########################################################
                if group["hist"]:
                    if group["transformer"] == "softplus":
                        denom_list.append(denom.reshape(1, -1).squeeze())
                        denom_inv_list.append((1 / denom).reshape(1, -1).squeeze())
                        m_v_eta.append(
                            (-step_size * torch.mul(exp_avg, (1 / denom)))
                            .reshape(1, -1)
                            .squeeze()
                        )
                    else:
                        denom_list.append(denom.reshape(1, -1).squeeze())
                        denom_inv_list.append(
                            (1 / (denom) ** (partial * 2)).reshape(1, -1).squeeze()
                        )
                        m_v_eta.append(
                            (
                                -step_size
                                * torch.mul(exp_avg, (1 / (denom) ** (partial * 2)))
                            )
                            .reshape(1, -1)
                            .squeeze()
                        )

        return {"denom": denom_list, "denom_inv": denom_inv_list, "m_v_eta": m_v_eta}
