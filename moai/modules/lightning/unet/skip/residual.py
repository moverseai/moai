import moai.nn.convolution as mic
import moai.nn.activation as mia

import torch
import omegaconf.omegaconf

__all__ = ['Residual']

class Residual(torch.nn.Module):
    def __init__(self,
        features:               int,
        blocks:                 int,
        convolution:            omegaconf.DictConfig,
        activation:             omegaconf.DictConfig,
        expansion:              int=2,
    ):
        super().__init__()
        self.convs = torch.nn.Sequential(*[
            mic.make_conv_block(block_type='conv2d',
                convolution_type=convolution.type,
                in_features=features, 
                out_features=features,
                activation_type=activation.type,
                convolution_params={
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'bias': False,
                },
                activation_params=activation.params
            ) for _ in range(blocks)
        ])
        self.shortcuts = torch.nn.Sequential(*[
            mic.make_conv_1x1(
                convolution_type='conv2d',
                in_channels=features,
                out_channels=features,
            ) for _ in range(blocks)
        ])
        self.activations = torch.nn.Sequential(*[
            mia.make_activation(
                activation_type=activation.type,
                features=features,
                **activation.get('params', {})
            ) for _ in range(blocks)
        ])

    def forward(self, 
        enc:    torch.Tensor,
        dec:    torch.Tensor,
        gate:   torch.Tensor=None,
    ) -> torch.Tensor:
        for c, s, a in zip(self.convs, self.shortcuts, self.activations):
            enc = c(enc)
            res = s(enc)
            enc = a(enc + res)        
        return torch.cat([dec, enc], dim=1)