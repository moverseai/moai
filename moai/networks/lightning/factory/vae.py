from moai.networks.lightning.feedforward import _create_processing_block

import typing
import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

import moai.networks.lightning as minet
import moai.nn.linear as mil

import torch
import inspect
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import toolz

from moai.modules.lightning.reparametrization.normal import NormalPrior

log = logging.getLogger(__name__)

__all__ = ["VariationalAutoencoder"]

class VariationalAutoencoder(minet.FeedForward):
    def __init__(self, 
        configuration:  omegaconf.DictConfig,
        io:             omegaconf.DictConfig,
        modules:        omegaconf.DictConfig,
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
        self.latent_dim = configuration.features_head.latent_dim
        self.repeat_val = configuration.repeat_val
        self.traversal_len = configuration.traversal_len
        self.traversal_step = configuration.traversal_step
        self.traversal_dim = configuration.traversal_dim
        self.traversal_init_value = configuration.traversal_init_value
        flatten = toolz.get_in(['flatten'], modules)        
        flatten = hyu.instantiate(flatten) if flatten else torch.nn.Flatten()
        prior = toolz.get_in(['reparametrization'], modules)        
        prior = hyu.instantiate(prior) if prior else NormalPrior()
        self.feature_head = Feature2MuStd(configuration, flatten)
        self.reparametrizer = Reparametrizer(prior) 
        self.encoder = hyu.instantiate(modules['encoder'])
        self.decoder = hyu.instantiate(modules['decoder'])
        self.generation = _create_processing_block(feedforward, "generation", monads=monads)
        self.enc_fwds, self.dec_fwds, self.f2mu_std_fwds, self.rep_fwds = [], [], [], []

        params = inspect.signature(self.encoder.forward).parameters
        enc_in = list(zip(*[mirtp.force_list(io.encoder[prop]) for prop in params]))
        enc_out = mirtp.split_as(mirtp.resolve_io_config(io.encoder['out']), enc_in)

        self.enc_res_fill = [mirtp.get_result_fillers(self.encoder, out) for out in enc_out]        
        get_enc_filler = iter(self.enc_res_fill)
        
        for keys in enc_in:
            self.enc_fwds.append(lambda td,
                tk=keys,
                args=params.keys(),
                enc=self.encoder,
                filler=next(get_enc_filler):
                    filler(td, enc(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            )

        params = inspect.signature(self.feature_head.forward).parameters
        rep_in = list(zip(*[mirtp.force_list(io.features_head[prop]) for prop in params]))
        rep_out = mirtp.split_as(mirtp.resolve_io_config(io.features_head['out']), enc_in)

        self.fhead_res_fill = [mirtp.get_result_fillers(self.feature_head, out) for out in rep_out]        
        get_fhead_filler = iter(self.fhead_res_fill)
        
        for keys in rep_in:
            self.f2mu_std_fwds.append(lambda td,
                tk=keys,
                args=params.keys(),
                rep=self.feature_head,
                filler=next(get_fhead_filler):
                    filler(td, rep(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            ) 

        params = inspect.signature(self.reparametrizer.forward).parameters
        rep_in = list(zip(*[mirtp.force_list(io.reparametrization[prop]) for prop in params]))
        rep_out = mirtp.split_as(mirtp.resolve_io_config(io.reparametrization['out']), enc_in)

        self.rep_res_fill = [mirtp.get_result_fillers(self.reparametrizer, out) for out in rep_out]        
        get_rep_filler = iter(self.rep_res_fill)
        
        for keys in rep_in:
            self.rep_fwds.append(lambda td,
                tk=keys,
                args=params.keys(),
                rep=self.reparametrizer,
                filler=next(get_rep_filler):
                    filler(td, rep(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            ) 

        params = inspect.signature(self.decoder.forward).parameters
        dec_in = list(zip(*[mirtp.force_list(io.decoder[prop]) for prop in params]))
        dec_out = mirtp.split_as(mirtp.resolve_io_config(io.decoder['out']), dec_in)

        self.dec_res_fill = [mirtp.get_result_fillers(self.decoder, out) for out in dec_out]
        get_dec_filler = iter(self.dec_res_fill)
        
        for keys in dec_in:
            self.dec_fwds.append(lambda td,
                tk=keys,
                args=params.keys(), 
                dec=self.decoder,
                filler=next(get_dec_filler):
                    filler(td, dec(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            )
            
    def decode(self,
        tensor: torch.Tensor
    ):
        return self.decoder(tensor)

    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for e, f, r, d in zip(self.enc_fwds, self.f2mu_std_fwds, self.rep_fwds, self.dec_fwds):
            e(td)
            f(td)
            r(td)
            d(td)
        return td

    def training_step_end(self, 
        train_outputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]
    ) -> None:
        if self.global_step and (self.global_step % self.visualization.interval == 0):
            self.visualization.visualizers[0](train_outputs['tensors'])
            self.visualization.visualizers[1](train_outputs['tensors'])
        return train_outputs['loss']


class Feature2MuStd(torch.nn.Module):
    def __init__(self,
        configuration:      omegaconf.DictConfig,
        flatten:            torch.nn.Module,
    ) -> None:
        super(Feature2MuStd, self).__init__()
        self.flatten = flatten
        
        self.linear_mu = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type=configuration.linear.activation.type,
            in_features=configuration.features_head.enc_out_dim,
            out_features=configuration.features_head.latent_dim
        )
        self.linear_logvar = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type=configuration.linear.activation.type,
            in_features=configuration.features_head.enc_out_dim,
            out_features=configuration.features_head.latent_dim
        )

    def forward(self,
        features:   torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x_flat = self.flatten(features)
        mu = self.linear_mu(x_flat)
        logvar = self.linear_logvar(x_flat)

        return mu, logvar,

class Reparametrizer(torch.nn.Module):
    def __init__(self,
        prior:            torch.nn.Module,
    )-> None:
        super(Reparametrizer, self).__init__()
        self.reparametrize = prior

    def forward(self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        return self.reparametrize(mu, logvar)