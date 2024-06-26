import logging
import types

import torch

from moai.core.model import MoaiLightningModule

log = logging.getLogger(__name__)

__all__ = ["Pretrained"]


class Pretrained(object):
    def __init__(self, filename: str, strict: bool = True):
        self.filename, self.strict = filename, strict

    def __call__(self, model: torch.nn.Module) -> None:
        log.info(f"Loading pretrained model weights from {self.filename}")
        checkpoint = torch.load(
            self.filename, map_location=lambda storage, loc: storage
        )
        # TODO: optimizer state + epoch and other metadata
        if "hparams" in checkpoint.keys():  # TODO: is 'hyper_parameters' now
            ckpt_hparams = checkpoint["hparams"]
            model.hparams = (
                ckpt_hparams
                if isinstance(ckpt_hparams, types.SimpleNamespace)
                else types.SimpleNamespace(**ckpt_hparams)
            )
        # TODO: add wrapper logic if needed
        # if isinstance(model, minet.Wrapper):
        #     model.load_state_dict(checkpoint['state_dict'], strict=self.strict)
        # else:
        #     model.load_state_dict(checkpoint['state_dict'], strict=self.strict)
        #     model.on_load_checkpoint(checkpoint) #NOTE: check why this is needed
        model.load_state_dict(checkpoint["state_dict"], strict=self.strict)
        model.on_load_checkpoint(checkpoint)  # NOTE: check why this is needed
