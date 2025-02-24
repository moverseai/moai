import logging
import typing

import toolz
import torch
import yaml
from pytorch_lightning.callbacks import Callback

from moai.core.execution.constants import Constants

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ["Config"]

log = logging.getLogger(__name__)


class Config(
    Callback,
    typing.Callable[
        [typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]],
        None,
    ],
):
    def __init__(
        self,
        path: str,
    ):
        self.path = path

    def on_fit_start(self, _: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        text = yaml.dump(pl_module.hparams[Constants._MOAI_][Constants._EXECUTION_])
        rr.log(
            f"{self.path}/execution",
            rr.TextDocument(text.strip(), media_type=rr.MediaType.TEXT),
            static=True,
        )
        extras = yaml.dump(
            toolz.dissoc(
                pl_module.hparams,
                Constants._MOAI_,
                Constants._DATA_,
                Constants._ENGINE_,
                Constants._MODULES_,
                Constants._PARAMETERS_,
                Constants._MONITORS_,
                Constants._MODEL_,
            )
        )
        rr.log(
            f"{self.path}/extras",
            rr.TextDocument(extras.strip(), media_type=rr.MediaType.TEXT),
            static=True,
        )

    def __call__(
        self,
        tensors: typing.Mapping[str, torch.Tensor],
    ) -> None:
        pass
