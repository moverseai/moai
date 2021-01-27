import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

import torch
import inspect
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging 

__all__ = ['Autoencoder']

log = logging.getLogger(__name__)

class Autoencoder(minet.FeedForward):
    def __init__(self, 
        configuration:  omegaconf.DictConfig, 
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
        super(Autoencoder, self).__init__(
            data=data, parameters=parameters,
            feedforward=feedforward, monads=monads,
            supervision=supervision, validation=validation,
            visualization=visualization, export=export,
        ) 
        self.encoder = hyu.instantiate(modules['encoder'])
        self.decoder = hyu.instantiate(modules['decoder'])
        self.enc_fwds, self.dec_fwds = [], []

        params = inspect.signature(self.encoder.forward).parameters
        enc_in = list(zip(*[mirtp.force_list(configuration.encoder[prop]) for prop in params]))
        enc_out = mirtp.split_as(mirtp.resolve_io_config(configuration.encoder['out']), enc_in)
        
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
        
        params = inspect.signature(self.decoder.forward).parameters
        dec_in = list(zip(*[mirtp.force_list(configuration.decoder[prop]) for prop in params]))
        dec_out = mirtp.split_as(mirtp.resolve_io_config(configuration.decoder['out']), dec_in)

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
        for e, d in zip(self.enc_fwds, self.dec_fwds):
            e(td)
            d(td)
        return td