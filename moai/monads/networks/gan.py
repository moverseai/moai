import logging
import typing

import hydra.utils as hyu
import toolz
import torch

log = logging.getLogger(__name__)

__all__ = ["GAN"]


class GAN(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str,
    ) -> None:
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu")
        hparams = ckpt["hyper_parameters"]
        model = toolz.dissoc(
            hparams["model"], "supervision", "validation", "feedforward"
        )
        self.model = hyu.instantiate(model)
        self.model.generator.load_state_dict(
            toolz.keymap(
                lambda s: s.replace("generator.", ""),
                toolz.keyfilter(
                    lambda s: s.startswith("generator."), ckpt["state_dict"]
                ),
            )
        )

        self.model.discriminator.load_state_dict(
            toolz.keymap(
                lambda s: s.replace("discriminator.", ""),
                toolz.keyfilter(
                    lambda s: s.startswith("discriminator."), ckpt["state_dict"]
                ),
            )
        )

    def forward(
        self,
        generate: typing.Optional[torch.Tensor] = None,
        discriminate: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if generate is not None:
            return self.model.generator(generate)
        if discriminate is not None:
            return self.model.discriminator(discriminate)
