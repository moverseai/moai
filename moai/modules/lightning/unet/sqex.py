from moai.nn.sqex import SqueezeExcite as SE

import torch
import omegaconf.omegaconf

__all__ = ['SqueezeExcite']

class SqueezeExcite(torch.nn.Module):
    def __init__(self,
        features:       int,
        squeeze:        omegaconf.DictConfig, # type: 'averageXd', # one of ['averageXd', 'attentionXd']
        excite:         omegaconf.DictConfig, # delta: activation_{type|params}, # ratio \in [0, 1], # operation_{type|params}: one of ['convXd', 'linear']
        mode:           str='mul',
        expansion:      int=2,
    ):
        super(SqueezeExcite, self).__init__()
        self.se = SE(
            features=features,
            squeeze=squeeze,
            excite=excite,
            mode=mode,
        )

    def forward(self, 
        enc:    torch.Tensor,
        dec:    torch.Tensor,
        gate:   torch.Tensor=None,
    ) -> torch.Tensor:
        se = self.se(enc)
        return torch.cat([dec, se], dim=1)