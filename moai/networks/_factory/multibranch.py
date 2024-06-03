import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

from collections import OrderedDict

import torch
import inspect
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging 
import toolz

__all__ = ['MultiBranch']

log = logging.getLogger(__name__)

def _create_processing_block(
    cfg: omegaconf.DictConfig, 
    attribute: str, 
    monads: omegaconf.DictConfig
):
    if not cfg and attribute in cfg:
        log.warning(f"Empty processing block ({attribute}) in feedforward model.")
    return hyu.instantiate(getattr(cfg, attribute), monads)\
        if cfg and attribute in cfg else torch.nn.Identity()

class MultiBranch(minet.FeedForward):
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
        super(MultiBranch, self).__init__(
            data=data, parameters=parameters,
            feedforward=feedforward, monads=monads,
            supervision=supervision, validation=validation,
            visualization=visualization, export=export,
        )
        self.encoder = hyu.instantiate(modules.encoder)
        self.decoders = torch.nn.ModuleDict(OrderedDict(
            (k, hyu.instantiate(d)) for k, d in sorted(
                toolz.dissoc(modules, 'encoder').items(), 
                key=lambda kvp: kvp[0]
            )
        ))
        self.enc_fwds = []

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

        self.dec_fwds = OrderedDict()
        for name, decoder in self.decoders.items():
            params = inspect.signature(decoder.forward).parameters
            dec_in = list(zip(*[mirtp.force_list(configuration[name][prop]) for prop in params if configuration[name][prop] is not None]))
            #NOTE: dec_in receives them ordered (list), and so args cannot be ignored (nulled) when not last in the list
            #TODO: fix it via mapping
            dec_out = mirtp.split_as(mirtp.resolve_io_config(configuration[name]['out']), dec_in)

            dec_res_fill = [mirtp.get_result_fillers(decoder, out) for out in dec_out]
            get_dec_filler = iter(dec_res_fill)
            
            self.dec_fwds[name] = []
            for keys in dec_in:
                self.dec_fwds[name].append(lambda td,
                    tk=keys,
                    args=params.keys(), 
                    dec=decoder,
                    filler=next(get_dec_filler):
                        filler(td, dec(**dict(zip(args,
                            list(
                                    td[k] if type(k) is str
                                    else list(td[j] for j in k)
                                for k in tk
                            )
                        ))))
                )
        self.pre_decode = torch.nn.ModuleList(
            _create_processing_block(feedforward, f'pre_decode{i}', monads)\
                for i in range(1, len(self.decoders) + 1)
        )

    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for e in self.enc_fwds:
            e(td)
        for ((k, v), pre) in zip(
            self.dec_fwds.items(),
            self.pre_decode,
        ):
            for d in v:
                td = pre(td)
                d(td)
        return td