import moai.nn.residual as mires
import moai.nn.sampling.spatial.downsample as mids
import moai.nn.convolution as mic
import moai.nn.utils as miu

import torch
import itertools
import typing
import logging
import omegaconf.omegaconf

log = logging.getLogger(__name__)

#NOTE: https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 
#NOTE: https://shuzhanfan.github.io/2018/11/ResNet/
#NOTE: https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d

__all__ = ["ResNet"]

class ResNet(torch.nn.Module):
    def __init__(self,
        configuration:  omegaconf.DictConfig, # in & out features, input
        preproc:        omegaconf.DictConfig, # blocks, type, convolution, activation
        residual:       omegaconf.DictConfig, # blocks, type, convolution, activation
        downscale:      omegaconf.DictConfig, # 
    ):
        super(ResNet, self).__init__()
        self.block_modules = torch.nn.ModuleDict() # build network
        in_features, out_features = configuration.in_features, configuration.start_features            
        preproc_blocks = torch.nn.ModuleList()
        for i in range(0, preproc.block.count): # preproc blocks
            preproc_blocks.append(mic.make_conv_block(
                block_type=preproc.block.type,
                convolution_type=preproc.convolution.type,
                in_features=in_features,
                out_features=out_features,
                activation_type=preproc.activation.type,                    
                activation_params=preproc.activation.params,
                convolution_params={ 
                    k: miu.repeat(preproc.block.count, v)[i] for k, v in preproc.convolution.params.items() 
                }
            ))
            in_features = out_features
            out_features *= 2
        self.block_modules.add_module('preproc', torch.nn.Sequential(*preproc_blocks))
        block_names = ["block" + str(i) for i in range(1, len(residual.block.setup))]
        block_names.append("features")
        out_features = int(in_features * residual.output_factor)
        bottleneck_features = int(in_features * residual.bottleneck_factor)
        for i, (block_name, block_size) in enumerate(zip(block_names, residual.block.setup)): # residual blocks
            residual_blocks = torch.nn.ModuleList()
            residual_blocks.append(mids.make_downsample( # 1 downscale operation
                downscale_type=downscale.type,
                features=in_features,
                **downscale.params
            ))
            for b in range(0, block_size):
                residual_blocks.append(mires.make_residual_block( # b residual connections
                    block_type=residual.block.type,
                    convolution_type=residual.convolution.type,
                    in_features=(in_features if b == 0 else out_features),
                    out_features=out_features,
                    bottleneck_features=bottleneck_features,
                    activation_type=residual.activation.type,
                    strided=(b == 0 and downscale.type == "in_block"),
                    convolution_params=residual.convolution.params,
                    activation_params=residual.activation.params,
                    downscale_params=downscale.params
                ))
            self.block_modules.add_module(block_name, torch.nn.Sequential(*residual_blocks))                
            in_features = out_features
            bottleneck_features *= 2
            out_features *= 2

    def forward(self, 
        data: torch.Tensor 
    ) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        features = [data]
        for _, block in self.block_modules.items():
            features.append(block(features[-1]))
        return features[-1], features[2:-1]