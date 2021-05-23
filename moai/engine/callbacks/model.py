from collections import UserList
from pytorch_lightning import Callback

import torch
import typing
import logging

log = logging.getLogger(__name__)

class ModelCallbacks(UserList):
    def __init__(self,
        list:       typing.Sequence[Callback]=None,
        model:      torch.nn.Module=None,
    ):
        super(ModelCallbacks, self).__init__(list)
        if model:
            self.data.extend((
                c for c in model.children() if isinstance(c, Callback)
            ))