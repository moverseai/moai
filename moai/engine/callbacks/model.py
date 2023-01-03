from collections import UserList
from pytorch_lightning import Callback
from moai.utils.engine.noop import NoOp

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
            if hasattr(model,"visualization"):
                if not isinstance(model.visualization,NoOp):
                    if isinstance(model.visualization.visualizers, typing.Sequence):
                        self.data.extend((
                            c for c in model.visualization.visualizers if isinstance(c, Callback)
                        ))
