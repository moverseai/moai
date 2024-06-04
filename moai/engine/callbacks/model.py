import logging
import typing
from collections import UserList

import torch
from pytorch_lightning import Callback

log = logging.getLogger(__name__)


class ModelCallbacks(UserList):
    def __init__(
        self,
        list: typing.Sequence[Callback] = None,
        model: torch.nn.Module = None,
    ):
        super().__init__(list)
        if model:
            self.data.extend((c for c in model.children() if isinstance(c, Callback)))
