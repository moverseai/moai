import logging
import typing

import omegaconf.omegaconf
import torch

import moai.nn.linear as mil

log = logging.getLogger(__name__)


class Basic(torch.nn.Module):
    """
    Responsible for predicting
    the mean and logvar of a normal distribution.
    """

    def __init__(
        self,
        mean_module: omegaconf.DictConfig,
        sigma_module: omegaconf.DictConfig,
    ) -> None:
        super(Basic, self).__init__()

        self.linear_mu = mil.make_linear_block(
            block_type=mean_module.type,
            linear_type="linear",
            activation_type=mean_module.activation.type,
            in_features=mean_module.in_features,
            out_features=mean_module.out_features,
        )

        self.linear_logvar = mil.make_linear_block(
            block_type=sigma_module.type,
            linear_type="linear",
            activation_type=sigma_module.activation.type,
            in_features=sigma_module.in_features,
            out_features=sigma_module.out_features,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> typing.Mapping[str, torch.Tensor]:
        mu = self.linear_mu(features)
        logvar = self.linear_logvar(features)

        # return mu, logvar
        return {
            "mu": mu,
            "logvar": logvar,
        }
