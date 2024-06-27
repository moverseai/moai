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
            if hasattr(model, "named_monitors"):
                # search within named monitors
                # for callbacks
                for monitor in model.named_monitors.values():
                    for oper in monitor.operations:
                        # get functions from operations
                        if isinstance(oper.func, Callback):
                            self.data.append(oper.func)
