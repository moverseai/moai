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
        model = toolz.dissoc(hparams['model'],'supervision', 'generation', 'validation')
        self.model = hyu.instantiate(model)
        self.model.encoder.load_state_dict(
            toolz.keymap(lambda s: s.replace('encoder.', ''), 
                toolz.keyfilter(
                    lambda s: s.startswith('encoder.'), 
                    ckpt['state_dict']
                )
            )
        )

        self.model.feature_head.load_state_dict(
            toolz.keymap(lambda s: s.replace('feature_head.', ''), 
                toolz.keyfilter(
                    lambda s: s.startswith('feature_head.'), 
                    ckpt['state_dict']
                )
            )
        )

        self.model.reparametrizer.load_state_dict(
            toolz.keymap(lambda s: s.replace('reparametrizer.', ''), 
                toolz.keyfilter(
                    lambda s: s.startswith('reparametrizer.'), 
                    ckpt['state_dict']
                )
            )
        )
        
        self.model.decoder.load_state_dict(
            toolz.keymap(lambda s: s.replace('decoder.', ''), 
                toolz.keyfilter(
                    lambda s: s.startswith('decoder.'), 
                    ckpt['state_dict']
                )
            )
        )

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
