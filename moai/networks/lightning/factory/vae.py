import moai.networks.lightning as minet
import moai.nn.convolution as mic
import moai.nn.deconvolution as mid
import moai.nn.linear as mil

import torch
import inspect
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import random
import numpy
import toolz

log = logging.getLogger(__name__)

__all__ = ["VariationalAutoencoder"]

class Repeater(torch.utils.data.Dataset):
    def __init__(self,
        z_dim: int,     
        repeat:     int=12,
        samples:    int=8,
    ):
        super(Repeater, self).__init__()
        self.samples = samples #[random.randint(0, max_size) for _ in range(samples)]
        self.repeat = repeat
        self.z_dim = z_dim
        self.total_z = torch.randn(self.samples, self.z_dim)

    def __len__(self) -> int:
        return self.repeat

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return self.total_z

class VariationalAutoencoder(minet.FeedForward):
    def __init__(self, 
        configuration:  omegaconf.DictConfig, # in & out features, input
        preproc:        omegaconf.DictConfig,
        residual:       omegaconf.DictConfig=None,
        downscale:      omegaconf.DictConfig=None,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        feedforward:    omegaconf.DictConfig=None,        
        monads:         omegaconf.DictConfig=None,        
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(VariationalAutoencoder, self).__init__(
            feedforward=feedforward, monads=monads, 
            visualization=visualization, data=data,
            supervision=supervision, validation=validation,
            export=export, parameters=parameters,
        )
        hidden_dim = configuration.hidden_dim
        latent_dim = configuration.latent_dim
        in_features = configuration.in_features
        encoder_blocks = torch.nn.ModuleList()
        for dim in hidden_dim:
            encoder_blocks.append(mic.make_conv_block(
                block_type="conv2d",
                convolution_type=preproc.convolution.type,
                in_features=in_features, 
                out_features=dim,
                activation_type=preproc.convolution.activation.type,
                convolution_params=preproc.convolution.params
            ))
            in_features = dim
        self.linear_mu = mil.make_linear_block(
            block_type=preproc.linear.type,
            linear_type="linear",
            activation_type = preproc.linear.activation.type,
            in_features=in_features * 4 * 4,
            out_features=latent_dim
        )
        self.linear_logvar = mil.make_linear_block(
            block_type=preproc.linear.type,
            linear_type="linear",
            activation_type = preproc.linear.activation.type,
            in_features=in_features * 4 * 4,
            out_features=latent_dim
        )
        self.decoder_input = mil.make_linear_block(
            block_type=preproc.linear.type,
            linear_type="linear",
            activation_type = preproc.linear.activation.type,
            in_features=latent_dim,
            out_features=hidden_dim[-1] * 4 * 4
        )
        hidden_dim.reverse()
        decoder_blocks = torch.nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            decoder_blocks.append(mid.make_deconv_block(
                block_type="deconv2d",
                deconvolution_type=preproc.deconvolution.type,
                in_features=hidden_dim[i], 
                out_features=hidden_dim[i + 1],
                activation_type=preproc.deconvolution.activation.type,
                deconvolution_params=preproc.deconvolution.params
            ))
        decoder_blocks.append(mid.make_deconv_block(
            block_type="deconv2d",
            deconvolution_type=preproc.deconvolution.type,
            in_features=hidden_dim[-1], 
            out_features=hidden_dim[-1],
            activation_type=preproc.deconvolution.activation.type,
            deconvolution_params=preproc.deconvolution.params
            ))
        decoder_blocks.append(mic.make_conv_block(
            block_type="conv2d",
            convolution_type=preproc.convolution.type,
            in_features=hidden_dim[-1], 
            out_features=configuration.out_features,
            activation_type="tanh",
            convolution_params={"kernel_size": 3, "stride": 1, "padding": 1, "output_padding": 1}
            ))
        self.encoder = torch.nn.Sequential(*encoder_blocks)
        self.decoder = torch.nn.Sequential(*decoder_blocks)
        self.input = configuration.input
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        #self.visualizer = visualization

    def encode(self,
        x: torch.Tensor
    ) -> [torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return [mu, logvar]

    def decode(self,
        z: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dim[0], 4, 4)
        x = self.decoder(x)
        return x

    def reparameterize(self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        x = td[self.input]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        td[f"mu"] = mu
        td[f"logvar"] = logvar
        td[f"latent"] = z
        td[f"generated"] = out
        return td

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        log.info(f"Instantiating ({self.data.val.iterator._target_.split('.')[-1]}) validation set data iterator")
        #val_iterator = hyu.instantiate(self.data.val.iterator)
        validation_loader = Repeater(self.latent_dim, repeat=4, samples=4)
        #validation_loader = hyu.instantiate(self.data.val.loader, val_iterator)
        return validation_loader

    def validation_step(self,
        batch: typing.Dict[str, torch.Tensor],
        batch_nb: int
    ) -> typing.Dict[str, torch.Tensor]:
        traversal = -3.0
        img_list = []
        for i in range(7):
            batch[:, 0] = traversal
            generated = self.decode(batch)
            img_list.append(generated)
            traversal += 1
        generated = torch.stack(img_list, 1)
        self.visualizer({'color_grid': generated}) #TODO repeater
        return generated

    def validation_epoch_end(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> None:
        pass
    
         

