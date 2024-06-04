import glob
import logging
import os
import typing

import toolz
import torch

from moai.monads.human.body.prior.human_body_prior import VPoser_v1 as VPoser

log = logging.getLogger(__name__)

__all__ = ["VPoser1"]


class VPoser1(typing.Callable[[torch.nn.Module], None]):
    def __init__(
        self,
        ckpt: str,
        cache: bool = False,
    ):
        if os.path.isdir(ckpt):
            self.filename = glob.glob(os.path.join(ckpt, "snapshots", "*.pt"))[-1]
        else:
            self.filename = ckpt
        if cache:
            self.state_dict = torch.load(
                self.filename, map_location=lambda storage, loc: storage
            )

    def _apply(self, module: torch.nn.Module) -> None:
        if isinstance(module, VPoser):
            state = (
                self.state_dict
                if hasattr(self, "state_dict")
                else torch.load(
                    self.filename, map_location=lambda storage, loc: storage
                )
            )
            module.load_state_dict(state, strict=True)
            log.info(f"Initialized VPoser1 from {self.filename}")

    def __call__(self, module: torch.nn.Module) -> None:
        module.apply(self._apply)
