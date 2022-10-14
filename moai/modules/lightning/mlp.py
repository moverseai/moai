import moai.nn.linear as mil

import torch
import logging
import omegaconf.omegaconf

log = logging.getLogger(__name__)

__all__ = ["MLP"]

class MLP(torch.nn.Module):
    def __init__(self,
        configuration: omegaconf.DictConfig,
    ) -> None:
        super(MLP, self).__init__()
        self.block_modules = torch.nn.ModuleDict()
        in_features, mid_features, out_features = configuration.in_features,\
                                                        configuration.hidden_features,\
                                                        configuration.out_features
        enc_list = torch.nn.ModuleList()

        for b in range(0, configuration.blocks):
            if b == 0:
                enc_list.append(mil.make_linear_block(
                    block_type=configuration.linear.type,
                    linear_type="linear",
                    in_features=in_features if b == 0  else mid_features,
                    out_features=mid_features,
                    activation_type=configuration.linear.activation.type
                ))
            else:
                enc_list.append(mil.make_linear_block(
                    block_type=configuration.linear.type,
                    linear_type="linear",
                    in_features=in_features if b == 0  else mid_features,
                    out_features=mid_features,
                    activation_type=configuration.linear.activation.type
                ))
        enc_list.append(mil.make_linear_block(
                block_type=configuration.prediction.type,
                linear_type="linear",
                in_features=mid_features,
                out_features=out_features,
                activation_type=configuration.prediction.activation.type
            ))
        self.sequential = torch.nn.Sequential(*enc_list)

    def forward(self,
        x:  torch.Tensor,
    ) -> torch.Tensor:
        return self.sequential(x)