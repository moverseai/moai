import torch
import logging
import omegaconf.omegaconf
import typing
from moai.modules.lightning.sampler.so3 import S2S2Mean, N0reparameterize

log = logging.getLogger(__name__)

class Homeomorphic(torch.nn.Module):
    """
    Responsible for predicting
    reparameterization parameters Rµ and σ respectively.
    Since σ are parameters of a distribution in R3,
    the corresponding network encσ does not pose any problems
    and can be chosen similarly as in classical VAE.

    Rµ could be derived from:
     - S2S1
     - S2S2
     - AlgebraMean
     - QuaternionMean
     - etc.

    sigma could be derived from:
      - N0reparameterize

    """

    def __init__(
        self,
        mean_module: omegaconf.DictConfig,
        sigma_module: omegaconf.DictConfig,
    ) -> None:
        super(Homeomorphic, self).__init__()
        self.mean_module = S2S2Mean(
            input_dims=mean_module.input_dims,
            output_dims=mean_module.output_dims
        )
        self.sigma_module = N0reparameterize(
            input_dim=sigma_module.input_dim,
            z_dim=sigma_module.z_dim,
            fixed_sigma=sigma_module.fixed_sigma
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_lie = self.mean_module(features)
        v, sigma = self.sigma_module(features)

        return mu_lie, v, sigma
