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
        self.latent_dim = configuration.reparametrization.latent_dim
        self.repeat_val = configuration.repeat_val
        self.traversal_len = configuration.traversal_len
        self.traversal_step = configuration.traversal_step
        self.traversal_dim = configuration.traversal_dim
        self.traversal_init_value = configuration.traversal_init_value
        flatten = toolz.get_in(['flatten'], modules)        
        flatten = hyu.instantiate(flatten) if flatten else torch.nn.Flatten()
        prior = toolz.get_in(['reparametrization'], modules)        
        prior = hyu.instantiate(prior) if prior else NormalPrior()
        self.reparemetrizer = Reparametrizer(configuration, prior, flatten) 
        self.encoder = hyu.instantiate(modules['encoder'])
        self.decoder = hyu.instantiate(modules['decoder'])
        self.enc_fwds, self.dec_fwds, self.rep_fwds = [], [], []

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

        params = inspect.signature(self.reparemetrizer.forward).parameters
        rep_in = list(zip(*[mirtp.force_list(io.reparametrization[prop]) for prop in params]))
        rep_out = mirtp.split_as(mirtp.resolve_io_config(io.reparametrization['out']), enc_in)

        self.rep_res_fill = [mirtp.get_result_fillers(self.reparemetrizer, out) for out in rep_out]        
        get_rep_filler = iter(self.rep_res_fill)
        
        for keys in rep_in:
            self.rep_fwds.append(lambda td,
                tk=keys,
                args=params.keys(),
                rep=self.reparemetrizer,
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

    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for e, d, r in zip(self.enc_fwds, self.dec_fwds, self.rep_fwds):
            e(td)
            r(td)
            d(td)
        return td

    def training_step_end(self, 
        train_outputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]
    ) -> None:
        if self.global_step and (self.global_step % self.visualization.interval == 0):
            self.visualization.visualizers[0](train_outputs['tensors'])
        # if self.global_step and (self.global_step % self.exporter.interval == 0):
        #     self.exporter(train_outputs['tensors'])
        return train_outputs['loss']

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass
        # log.info(f"Instantiating ({self.data.val.iterator._target_.split('.')[-1]}) validation set data iterator")
        # validation_loader = Repeater(self.latent_dim, self.repeat_val)
        # return validation_loader

    def validation_step(self,
        batch: typing.Dict[str, torch.Tensor],
        batch_nb: int
    ) -> typing.Dict[str, torch.Tensor]:
        pass
        # traversal_value = self.traversal_init_value
        # img_list = []
        # for i in range(self.traversal_len):
        #     if self.traversal_dim < 0:
        #         batch[:] = traversal_value
        #     else:
        #         batch[self.traversal_dim] = traversal_value
        #     generated = self.decode(batch)
        #     img_list.append(generated)
        #     traversal_value += self.traversal_step
        # generated = torch.stack(img_list, 1)
        # return generated.squeeze(0)

    def validation_epoch_end(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> None:
        pass
        # self.visualizer.latent_visualizers[0](torch.stack(tensors, 1))


class Reparametrizer(torch.nn.Module):
    def __init__(self,
        configuration:    omegaconf.DictConfig,
        prior:            torch.nn.Module,
        flatten:          torch.nn.Module,
    )-> None:
        super(Reparametrizer, self).__init__()
        self.reparametrize = prior
        self.flatten = flatten
        self.linear_mu = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type = configuration.linear.activation.type,
            in_features=configuration.reparametrization.enc_out_dim,
            out_features=configuration.reparametrization.latent_dim
        )
        self.linear_logvar = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type = configuration.linear.activation.type,
            in_features=configuration.reparametrization.enc_out_dim,
            out_features=configuration.reparametrization.latent_dim
        )
        self.linear_to_dec = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type = configuration.linear.activation.type,
            in_features=configuration.reparametrization.latent_dim,
            out_features=configuration.reparametrization.enc_out_dim
        )

    def forward(self,
        features:   torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = self.flatten(features)
        mu = self.linear_mu(x_flat)
        logvar = self.linear_logvar(x_flat)
        z = self.reparametrize(mu, logvar)
        z = self.linear_to_dec(z)
        z_reshaped = z.reshape_as(features)

        return z_reshaped, mu, logvar


class Repeater(torch.utils.data.Dataset):
    def __init__(self,
        z_dim:   int,     
        repeat:  int
    ):
        super().__init__()
        self.repeat = repeat
        self.z_dim = z_dim
        self.total_z = torch.randn(self.repeat, self.z_dim)

    def __len__(self) -> int:
        return self.repeat

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.total_z[index]