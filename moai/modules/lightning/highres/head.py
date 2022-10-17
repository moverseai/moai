from moai.monads.utils.spatial import spatial_dim_list

import moai.nn.convolution as mic
import moai.nn.residual as mires
import moai.nn.deconvolution as midec 
import moai.nn.utils as miu
import moai.nn.sampling.spatial.downsample as mids
import moai.nn.sampling.spatial.upsample as mius

import toolz
import torch
import typing
import functools
import omegaconf.omegaconf

__all__ = ["TopBranch", "AllBranches","Higher"]

class TopBranch(torch.nn.Module):
    def __init__(self, 
        stages:         int,
        start_features: int,
        out_features:   int,
        convolution:    str="conv2d",
        activation:     str="none",
        kernel_size:    int=1,
        padding:        int=1,
        inplace:        bool=True
    ):
        super(TopBranch, self).__init__()
        self.conv = mic.make_conv_block(
            block_type="conv2d",
            convolution_type=convolution,
            in_features=start_features,
            out_features=out_features,
            activation_type=activation,
            convolution_params={
                'kernel_size': kernel_size,
                'padding': padding                
            },
            activation_params={
                'inplace': inplace
            }
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, 
        all_branches: typing.Union[typing.Tuple[torch.Tensor, ...], typing.List[torch.Tensor]]
    ) -> torch.Tensor:
        return self.conv(all_branches[0])

class AllBranches(torch.nn.Module):
    def __init__(self, 
        stages:         int,
        start_features: int,
        out_features:   int,
        projection:     omegaconf.DictConfig,
        prediction:     omegaconf.DictConfig,
        upsample:       omegaconf.DictConfig
    ):
        super(AllBranches, self).__init__()
        in_features = sum((start_features * (2 ** i) for i in range(stages)))
        self.projection = mic.make_conv_block(
            block_type='conv2d',
            convolution_type=projection.convolution,
            in_features=in_features,
            out_features=in_features,
            activation_type=projection.activation,
            convolution_params={
                'kernel_size': projection.kernel_size,
                'padding': projection.padding,
            },
            activation_params={
                'inplace': projection.inplace
            }
        )
        self.prediction = mic.make_conv_block(
            block_type='conv2d',
            convolution_type=prediction.convolution,
            in_features=in_features,
            out_features=out_features,
            activation_type=prediction.activation,
            convolution_params={
                'kernel_size': prediction.kernel_size,
                'padding': prediction.padding,
            },
            activation_params={
                'inplace': prediction.inplace
            }
        )
        self.upsample = functools.partial(torch.nn.functional.interpolate,
            mode=upsample.mode
        )
        
    def forward(self, 
        all_branches: typing.Union[typing.Tuple[torch.Tensor, ...], typing.List[torch.Tensor]]
    ) -> torch.Tensor:
        top = all_branches[0]
        dims = spatial_dim_list(top)
        up = (self.upsample(x, size=dims) for x in all_branches[1:])
        concat = torch.cat([top, *up], dim=1)
        return self.prediction(self.projection(concat))


class Higher(torch.nn.Module):
    def __init__(
        self,
        stages: int,
        start_features: int, #in channels
        out_features: int, #initial out channels
        deconvolution_modules: int, #nr of deconvolution modules
        deconvolution: omegaconf.DictConfig,
        residual:   omegaconf.DictConfig,
        final: omegaconf.DictConfig,
        aggregator: omegaconf.DictConfig,
    ):
        super(Higher, self).__init__()

        self.deconvolution_modules = deconvolution_modules
        self.deconvolution = deconvolution

        start_features_ = lambda i : start_features if i == 0 else deconvolution.deconv_out_features
        self.deconv_layers = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                midec.make_deconv_block(
                    block_type = deconvolution.block,
                    in_features = (start_features_(i) + out_features) \
                                 if deconvolution.concat \
                                 else start_features_(i),
                    out_features=deconvolution.deconv_out_features,
                    deconvolution_type = deconvolution.type,
                    activation_type=deconvolution.activation,
                    deconvolution_params={
                        'kernel_size': deconvolution.kernel_size,
                        'padding': deconvolution.padding,
                        'output_padding': deconvolution.output_padding,
                        'stride': 2,
                    }),
                torch.nn.Sequential(*[
                    mires.make_residual_block(
                        block_type=residual.type,
                        convolution_type=residual.convolution,
                        in_features=deconvolution.deconv_out_features,
                        out_features=deconvolution.deconv_out_features,
                        bottleneck_features=residual.bottleneck_features,
                        activation_type=residual.activation,
                        activation_params=residual.activation.params or { 'inplace': True },
                        strided=False,
                        convolution_params=toolz.merge(residual.convolution.params or {}, {
                                'bias':         False,
                        })) for r in range(deconvolution.residual_units)
                ])
            ]) for i in range(deconvolution_modules)
        ])

        #final layer
        self.final_layers = torch.nn.ModuleList([
            mic.make_conv_block(
                block_type="conv2d",
                convolution_type=final.convolution,
                in_features=start_features,
                out_features=out_features,
                activation_type=final.activation,
                convolution_params={
                    'kernel_size': final.kernel_size,
                    'padding': final.padding,
                }
            ),
            torch.nn.ModuleList([
                mic.make_conv_block(
                    block_type="conv2d",
                    convolution_type=final.convolution,
                    in_features=deconvolution.deconv_out_features,
                    out_features=out_features,
                    activation_type=final.activation,
                    convolution_params={
                        'kernel_size': final.kernel_size,
                        'padding': final.padding,
                    }
                ) for i in range(deconvolution_modules)
            ])
        ])

        self.aggregator_func = functools.partial(
            torch.nn.functional.interpolate,
            size=(aggregator.height,aggregator.width),
            mode=aggregator.mode,
            align_corners=aggregator.align_corners,
            recompute_scale_factor = aggregator.recompute_scale_factor
        )
    
    def forward(self, 
        all_branches: typing.Union[typing.Tuple[torch.Tensor, ...], typing.List[torch.Tensor]]
    ) -> torch.Tensor:
        
        isTraining = self.final_layers[0].training
        x = all_branches[0]
        y = self.final_layers[0](x) #output from top branch
        final_outputs = []
        if isTraining:
            final_outputs.append(y)
        else:
            heatmaps_avg = []
            y_upscaled = self.aggregator_func(y)
            heatmaps_avg.append(y_upscaled)
        for i in range(self.deconvolution_modules):
            if self.deconvolution.concat:
                x = torch.cat((x, y), 1)
            x = self.deconv_layers[i](x)
            y = self.final_layers[1][i](x)
            if isTraining:
                final_outputs.append(y)
            else:
                #upscale output to be used in the aggregator
                y_upscaled = self.aggregator_func(y)
                heatmaps_avg.append(y_upscaled)
        
        if not isTraining:
            final_outputs.append(
                torch.mean(torch.stack(heatmaps_avg), dim = 0)
            )
            final_outputs.append(torch.zeros((1)))
        
        return final_outputs