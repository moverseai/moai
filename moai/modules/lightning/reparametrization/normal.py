from moai.utils.arguments import assert_numeric

import torch
import logging

log = logging.getLogger(__name__)

__all__ = ["NormalPrior"]

class NormalPrior(torch.nn.Module):
    def __init__(self,
        mode:       str='standard', # ['standard', 'softplus']
        beta:       float=1.0,
        threshold:  float=20.0
    ) -> None:
        super(NormalPrior, self).__init__()
        assert_numeric(log, 'beta', beta, 0.0, None)
        assert_numeric(log, 'threshold', threshold, 0.0, None)
        self.softplus = torch.nn.Softplus(beta, threshold)
        self.prior= self.prior_with_softplus if mode == 'softplus'\
                        else self.prior_with_eps

    def prior_with_eps(self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def prior_with_softplus(self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.distributions.normal.Normal(
            mu,
            self.softplus(logvar)
        )
        return z.rsample()

    def forward(self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return self.prior(mu, logvar)