import itertools
import moai.networks.lightning as minet
import moai.nn.convolution as mic
import moai.nn.sampling.spatial.downsample as mids
import moai.nn.sampling.spatial.upsample as mius

import torch
import hydra.utils as hyu
import omegaconf.omegaconf as omegaconf
import typing
import logging
import toolz

log = logging.getLogger(__name__)

__all__ = ["UNet"]

class UNet(minet.FeedForward):
    class DefaultSkip(torch.nn.Module):
        def __init__(self):
            super(UNet.DefaultSkip, self).__init__()

        def forward(self, 
            enc:    torch.Tensor,
            dec:    torch.Tensor,
            gate:   torch.Tensor=None,
        ) -> torch.Tensor:
            return torch.cat([dec, enc], dim=1)

    def __init__(self,
        configuration:  omegaconf.DictConfig,
        modules:        omegaconf.DictConfig=None,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        feedforward:    omegaconf.DictConfig=None,
        monads:         omegaconf.DictConfig=None,
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(UNet, self).__init__(
            data=data, parameters=parameters,
            feedforward=feedforward, monads=monads,
            supervision=supervision, validation=validation,
            export=export, visualization=visualization,            
        )        
        # Encoder
        convolution = configuration.encoder.convolution
        activation = configuration.encoder.activation
        downscale = configuration.encoder.downscale
        block_features = configuration.block_features
        down_block_convs = configuration.encoder.blocks
        in_features = configuration.in_features
        self.enc = torch.nn.ModuleDict()
        self.down = torch.nn.ModuleDict()
        for f, b in zip(block_features, down_block_convs):
            enc_block = torch.nn.Sequential()
            for i in range(b): #TODO: can modify encoder conv block
                enc_block.add_module(f'enc{f}_{i}', mic.make_conv_block(
                    block_type='conv2d',
                    convolution_type=convolution.type,
                    in_features=in_features if i == 0 else f, 
                    out_features=f,
                    activation_type=activation.type,
                    convolution_params=convolution.params,
                    activation_params=activation.params
                ))
            in_features = f
            self.enc.add_module(f'enc_block{f}', enc_block)            
            self.down.add_module(f'down{f}', mids.make_downsample(
                downscale_type=downscale.type,
                features=f,
                kernel_size=3 if downscale.type == 'maxpool2d_aa' else 2,
                **(downscale.params if downscale.params is not None else {})
            ))        
        # Bottleneck
        bottleneck = configuration.bottleneck
        self.bottleneck = torch.nn.Sequential()
        for i in range(bottleneck.blocks): #TODO: can modify bottleneck block
            self.bottleneck.add_module(f'bottleneck{i}', mic.make_conv_block(
                block_type='conv2d',
                convolution_type=bottleneck.convolution.type,
                in_features=block_features[-1] if i == 0 else bottleneck.features, 
                out_features=bottleneck.features,
                activation_type=bottleneck.activation.type,
                convolution_params=bottleneck.convolution.params,
                activation_params=bottleneck.activation.params
            ))
        gate = toolz.get_in(['gate'], modules)        
        self.gate = hyu.instantiate(gate, bottleneck.features) if gate else torch.nn.Identity()        
        # Decoder
        convolution = configuration.decoder.convolution
        activation = configuration.decoder.activation
        upscale = configuration.decoder.upscale
        skip = toolz.get_in(['skip'], modules)
        expansion = toolz.get_in(['skip', 'expansion'], modules) or 2.0        
        up_block_convs = configuration.decoder.blocks
        up_block_convs = up_block_convs or reversed(down_block_convs)
        self.dec = torch.nn.ModuleDict()
        self.up = torch.nn.ModuleDict()
        self.skip = torch.nn.ModuleDict()
        skip_params = configuration.skip or {}
        if not isinstance(skip_params, typing.Sized) or len(skip_params) <= 1:
            skip_params = list(itertools.repeat(skip_params, len(down_block_convs)))
        for f, b, sp in zip(reversed(block_features), up_block_convs, skip_params):  #TODO: can modify upscale block
            self.up.add_module(f'up{f}', torch.nn.Sequential(
                mius.make_upsample(
                    upscale_type=upscale.type,
                    features=f * 2,
                    **(upscale.params if upscale.params is not None else {}),
                ),
                mic.make_conv_1x1(
                    convolution_type=convolution.type,
                    in_channels=f * 2, out_channels=f,
                ) if upscale.type == 'upsample2d' else torch.nn.Identity()
            ))#TODO: need to downscale features too
            #NOTE: one case is deconv2d and the other is upscale and project
            #NOTE: we now only use upscale and project
            self.skip.add_module(f'skip{f}', 
                hyu.instantiate(skip, f, **sp) if skip is not None else UNet.DefaultSkip()
            )
            dec_block = torch.nn.Sequential()
            skip_features = int(f * expansion)
            for i in range(b):
                dec_block.add_module(f'dec{f}_{i}', mic.make_conv_block(
                    block_type='conv2d',
                    convolution_type=convolution.type,
                    in_features=skip_features if i == 0 else f, 
                    out_features=f,
                    activation_type=activation.type,
                    convolution_params=convolution.params,
                    activation_params=activation.params
                ))
            self.dec.add_module(f'dec_block{f}', dec_block)        
        # Prediction
        prediction = configuration.prediction
        out_features = configuration.out_features
        self.pred = mic.make_conv_block('conv2d',
            convolution_type=prediction.convolution.type,
            in_features=block_features[0], out_features=out_features,
            activation_type=prediction.activation.type,
            convolution_params=prediction.convolution.params,
            activation_params=prediction.activation.params,
        )
        self.input = configuration.input
        self.output = configuration.output
        
    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for i, o in zip(self.input, self.output):
            data = td[i]
            features = [data]
            skipped = []
            for e, d in zip(self.enc.values(), self.down.values()):
                skipped.append(e(features[-1]))
                features.append(d(skipped[-1]))
            bottleneck = self.bottleneck(features[-1])
            skipped.reverse()
            features = [bottleneck]
            gate = self.gate(bottleneck)
            for i, (u, s, d) in enumerate(zip(
                self.up.values(),
                self.skip.values(),
                self.dec.values()
            )):
                up = u(features[-1])
                skip = s(skipped[i], up, gate)
                features.append(d(skip))
            out = self.pred(features[-1])
            td[o] = out
        return td