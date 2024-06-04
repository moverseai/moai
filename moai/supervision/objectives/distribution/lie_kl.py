import math

import numpy as np
import torch

__all__ = ["LieKL"]


class LieKL(torch.nn.KLDivLoss):
    def __init__(
        self,
        is_input_log: bool = False,
        is_target_log: bool = False,
    ):
        super(LieKL, self).__init__(reduction="none", log_target=is_target_log)
        self.is_input_log = is_input_log

    def log_posterior(self, v, sigma, k=10):
        theta = v.norm(p=2, dim=-1, keepdim=True) + 1e-6  # [n,B,1]
        u = v / theta  # [n,B,3]

        angles = (
            2 * math.pi * torch.arange(-k, k + 1, device=u.device, dtype=v.dtype)
        )  # [2k+1]

        theta_hat = theta[..., None, :] + angles[:, None]  # [n,B,2k+1,1]

        clamp = 1e-6
        x = u[..., None, :] * theta_hat  # [n,B,2k+1,3]
        # x should be possitive as should ender a log function

        # [n,(2k+1),B,3] or [n,(2k+1),B]
        # log_p = self.reparameterize._log_posterior(x.permute([0, 2, 1, 3]).contiguous())
        min_sigma = 1e-6
        sigma = torch.clamp(sigma, min=min_sigma)
        log_p = (
            torch.distributions.normal.Normal(
                torch.zeros_like(sigma),
                sigma,
            )
            .log_prob(x.permute([0, 2, 1, 3]))
            .sum(-1)
            .contiguous()
        )

        if len(log_p.size()) == 4:
            log_p = log_p.sum(-1)  # [n,(2k+1),B]

        log_p = log_p.permute([0, 2, 1])  # [n,B,(2k+1)]

        theta_hat_squared = torch.clamp(theta_hat**2, min=clamp)

        log_p.contiguous()
        cos_theta_hat = torch.cos(theta_hat)

        # [n,B,(2k+1),1]
        log_vol = torch.log(
            theta_hat_squared / torch.clamp(2 - 2 * cos_theta_hat, min=clamp)
        )

        log_p = log_p + log_vol.sum(-1)
        log_p = logsumexp(log_p, -1)

        return log_p

    def log_prior(self, z):
        prior = torch.tensor([-np.log(8 * (np.pi**2))], device=z.device)
        return prior.expand_as(z[..., 0])

    def forward(self, v, sigma, z, k=10):
        log_q_z_x = self.log_posterior(v, sigma, k)
        log_p_z = self.log_prior(z)
        kl = log_q_z_x - log_p_z
        return kl


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
