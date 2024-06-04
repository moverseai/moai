import typing

import torch

__all__ = ["AdamT"]

# NOTE: from https://github.com/xuebin-zh/AdamT
# NOTE: described in https://arxiv.org/pdf/2001.06130.pdf


class AdamT(torch.optim.Optimizer):
    """Implements the AdamT algorithm.

    - **Paper**: [ADAMT: A Stochastic Optimization with Trend Correction Scheme](https://arxiv.org/pdf/2001.06130.pdf).
    - **Implementation**: [GitHub @ xuebin-zh](https://github.com/xuebin-zh/AdamT).

    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: typing.Tuple[float, float, float, float] = (0.9, 0.999, 0.9, 0.999),
        phis: typing.Tuple[float, float] = (1, 1),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= betas[3] < 1.0:
            raise ValueError("Invalid beta parameter at index 3: {}".format(betas[3]))
        if not 0.0 <= phis[0] <= 100.0:
            raise ValueError("Invalid phi parameter at index 0: {}".format(phis[0]))
        if not 0.0 <= phis[1] <= 100.0:
            raise ValueError("Invalid phi parameter at index 1: {}".format(phis[1]))
        defaults = dict(
            lr=lr, betas=betas, phis=phis, eps=eps, weight_decay=weight_decay
        )
        super(AdamT, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamT, self).__setstate__(state)

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
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    # First-order
                    # state["m"] = grad
                    state["m"] = torch.zeros_like(p.data)
                    # First-order level information
                    # state["L_m"] = grad
                    state["L_m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    # Second-order
                    # state["v"] = grad * grad
                    state["v"] = torch.zeros_like(p.data)
                    # Second-order level information
                    # state["L_v"] = grad * grad
                    state["L_v"] = torch.zeros_like(p.data)
                    # Holt's linear trend information for first-order
                    state["B_m"] = torch.zeros_like(p.data)
                    # Holt's linear trend information for second-order
                    state["B_v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                L_m, L_v, B_m, B_v = (
                    state["L_m"],
                    state["L_v"],
                    state["B_m"],
                    state["B_v"],
                )
                beta1, beta2, beta3, beta4 = group["betas"]
                phi1, phi2 = group["phis"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Partially update first-order trend information
                B_m.mul_(beta3 * phi1).add_((beta3 - 1), L_m)
                # Partially update second-order trend information
                B_v.mul_(beta4 * phi2).add_((beta4 - 1), L_v)
                # Fully update first-order level information
                torch.add(beta1 * m, (1 - beta1) * grad, out=L_m)
                # Fully update second-order level information
                torch.addcmul(beta2 * v, (1 - beta2), grad, grad, out=L_v)
                # Fully update first-order trend information
                B_m.add_((1 - beta3), L_m)
                # Fully update second-order trend information
                B_v.add_((1 - beta4), L_v)
                # Update first-order and second-order
                torch.add(L_m, phi1 * B_m, out=m)
                torch.add(L_v, phi2 * B_v, out=v)
                # Bias correction
                # m_hat = torch.div(L_m, (1 - beta1 ** state["step"])) + \
                #         torch.div(phi1 * B_m, (1 - beta3 ** state["step"]))
                # v_hat = torch.div(L_v, (1 - beta2 ** state["step"])) + \
                #         torch.div(phi2 * B_v, (1 - beta4 ** state["step"]))
                m_hat = torch.div(L_m, (1 - beta1 ** state["step"])) + torch.div(
                    B_m,
                    (
                        (1 - beta3)
                        * torch.div(
                            1 - (beta3 * phi1) ** state["step"], 1 - beta3 * phi1
                        )
                    ),
                )

                v_hat = torch.div(L_v, (1 - beta2 ** state["step"])) + torch.div(
                    B_v,
                    (
                        (1 - beta4)
                        * torch.div(
                            1 - (beta4 * phi2) ** state["step"], 1 - beta4 * phi2
                        )
                    ),
                )
                # Update the parameters

                # if len(torch.nonzero(torch.isnan(torch.div(m_hat, (v_hat.sign() * torch.sqrt(v_hat.abs()) + group["eps"]))))) != 0:
                # print(torch.div(m_hat, (v_hat.sign() * torch.sqrt(v_hat.abs()) + group["eps"])))
                # print(v_hat.sign() * torch.sqrt(v_hat.abs()))
                # print(torch.div(m_hat, (v_hat.sign() * torch.sqrt(v_hat.abs()) + group["eps"])))
                # print(torch.div(m_hat, (v_hat.sign() * torch.sqrt(v_hat.abs()) + group["eps"])))
                # p.data.add_(-group["lr"], torch.div(m_hat, (v_hat.sign() * torch.sqrt(v_hat.abs()) + group["eps"])))
                p.data.add_(
                    -group["lr"],
                    torch.div(m_hat, (torch.sqrt(v_hat.abs()) + group["eps"])),
                )
                # print(v_hat.sign())
                # p.data.add_(-group["lr"], torch.div(m_hat, (v_hat.sign() * torch.sqrt(v_hat.abs() + group["eps"]))))

        return loss
