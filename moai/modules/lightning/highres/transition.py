import moai.nn.convolution as mic
import moai.nn.residual as mires
import moai.nn.utils as miu
import moai.nn.sampling.spatial.downsample as mids
import moai.nn.sampling.spatial.upsample as mius

import torch
import typing
import omegaconf.omegaconf

__all__ = ["StartTransition", "StageTransition"]

class StartTransition(torch.nn.Module):
    def __init__(self, 
        in_features:           int,
        start_features:        int,
        identity:               omegaconf.DictConfig,
        branched:               omegaconf.DictConfig,
    ):
        super(StartTransition, self).__init__()        
        self.conv_id = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=identity.convolution,
            in_features=in_features,
            out_features=start_features,
            activation_type=identity.activation,
            convolution_params={
                'kernel_size':  identity.kernel_size,
                'stride':       identity.stride,
                'padding':      identity.padding
            }
        )
        self.downscale_br = mids.make_downsample(
            downscale_type=branched.downscale,
            features=in_features,
            kernel_size=3 if branched.downscale == 'maxpool2d_aa' else 2,
        )
        self.conv_br = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=branched.convolution,
            in_features=in_features,
            out_features=start_features * 2,
            activation_type=branched.activation,
            convolution_params={
                'kernel_size':  branched.kernel_size,
                'stride':       branched.stride,
                'padding':      branched.padding
            }
        )

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        return [self.conv_id(x), self.conv_br(self.downscale_br(x))]

class StageTransition(torch.nn.Module):
    def __init__(self, 
        branches:               int,
        prev_branch_features:   int,        
        branched:               omegaconf.DictConfig,
    ):
        super(StageTransition, self).__init__()
        self.list = torch.nn.ModuleList([
            *(torch.nn.Identity() for _ in range(branches)), 
            torch.nn.Sequential(
                mids.make_downsample(
                    downscale_type=branched.downscale,
                    features=prev_branch_features,
                    kernel_size=3 if branched.downscale == 'maxpool2d_aa' else 2,
                ),
                mic.make_conv_block(
                    block_type="conv2d",
                    convolution_type=branched.convolution,
                    in_features=prev_branch_features,
                    out_features=prev_branch_features * 2,
                    activation_type=branched.activation,
                    convolution_params={
                        'kernel_size':  branched.kernel_size,
                        'stride':       branched.stride,
                        'padding':      branched.padding
                    },
                )
            )
        ])

    def forward(self, 
        branches: typing.Union[typing.Tuple[torch.Tensor, ...], typing.List[torch.Tensor]]
    ) -> typing.List[torch.Tensor]:
        return [m(t) for t, m in zip(branches + [branches[-1]], self.list)]