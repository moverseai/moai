import moai.nn.residual as mires
import moai.nn.sampling.spatial.downsample as mids
import moai.nn.sampling.spatial.upsample as mius
import omegaconf.omegaconf

import torch

#NOTE: from https://github.com/anibali/pytorch-stacked-hourglass/blob/master/src/stacked_hourglass/model.py
#NOTE: from https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/posenet.py

__all__ = ["Hourglass"]

class Hourglass(torch.nn.Module):
    def __init__(self, 
        convolution:        omegaconf.DictConfig,
        activation:         omegaconf.DictConfig,
        downscale:          str="maxpool2d",
        upscale:            str="upsample2d",
        residual:           str="preactiv_bottleneck",
        depth:              int=4,
        features:           int=128,
    ):
        super(Hourglass, self).__init__()
        new_features = features
        self.up1 = mires.make_residual_block(
            block_type=residual,
            convolution_type=convolution.type,
            in_features=features,
            out_features=features,
            bottleneck_features=features,
            activation_type=activation.type,
            strided=False,
        )
        # Lower branch        
        self.pool1 = mids.make_downsample(
            downscale_type=downscale,
            features=features,
            kernel_size=3 if downscale == 'maxpool2d_aa' else 2,
        )
        self.low1 = mires.make_residual_block(
            block_type=residual,
            convolution_type=convolution.type,
            in_features=features,
            out_features=new_features,
            bottleneck_features=new_features,
            activation_type=activation.type,
            strided=False,
        )
        self.depth = depth
        # Recursive hourglass        
        self.low2 = Hourglass(
                depth=depth-1, 
                features=new_features,
                upscale = upscale,#added
                convolution=convolution,
                activation=activation,
                downscale=downscale,
                residual=residual,        
            ) if self.depth > 1 \
            else mires.make_residual_block(
                block_type=residual,
                convolution_type=convolution.type,
                in_features=new_features,
                out_features=new_features,
                bottleneck_features=new_features,
                activation_type=activation.type,
                strided=False,
            )
        self.low3 = mires.make_residual_block(
                block_type=residual,
                convolution_type=convolution.type,
                in_features=new_features,
                out_features=features,
                bottleneck_features=features,
                activation_type=activation.type,
                strided=False,
            )
        self.up2 = mius.make_upsample(
            upscale_type=upscale,
            features=features,
            mode="bilinear" #NOTE or nearest? or extract as config param?
        )

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2