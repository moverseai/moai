import toolz
import torch
import hydra.utils as hyu
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['VAE']

class VAE(torch.nn.Module):
    def __init__(self,
        checkpoint:       str,
    ) -> None:
        super().__init__()
        ckpt = torch.load(checkpoint, map_location='cpu')
        hparams = ckpt['hyper_parameters']
        model = toolz.dissoc(hparams['model'], 'supervision', 'validation', 'monads', 'parameters', 'feedforward')
        self.model = hyu.instantiate(model)
        self.model.encoder.load_state_dict(
            toolz.keymap(lambda s: s.replace('encoder.', ''), 
                toolz.keyfilter(
                    lambda s: 'encoder.' in s, 
                    ckpt['state_dict']
                )
            )
        )
        self.model.decoder.load_state_dict(
            toolz.keymap(lambda s: s.replace('decoder.', ''), 
                toolz.keyfilter(
                    lambda s: 'decoder.' in s, 
                    ckpt['state_dict']
                )
            )
        )
        #TODO load feature_head and reparam modules weights

    def forward(self,
        encode:     typing.Optional[torch.Tensor]=None,
        decode:     typing.Optional[torch.Tensor]=None,
        autoencode:     typing.Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        if encode is not None:            
            return self.model.encoder(encode)
        if decode is not None:
            return self.model.decoder(decode)
        if autoencode is not None:
            return self.model.decoder(self.reparametrizer(self.feature_head(self.model.encoder(autoencode))))
