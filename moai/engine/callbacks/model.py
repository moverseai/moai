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
            if hasattr(model, "named_flows"):  # check for monads in the model
                for flow in model.named_flows.values():
                    for module in flow.values():
                        if isinstance(module, Callback):
                            self.data.append(module)

            if hasattr(model, "named_components"):
                for component in model.named_components.values():
                    if isinstance(component, Callback):
                        self.data.append(component)
            if hasattr(model, "named_monitors"):
                # search within named monitors
                # for callbacks
                for monitor in model.named_monitors.values():
                    for oper in monitor.operations:
                        # get functions from operations
                        if isinstance(oper.func, Callback):
                            self.data.append(oper.func)
            if hasattr(model, "named_objectives"):
                for stage in model.named_objectives.values():
                    for obj in stage.values():
                        if isinstance(obj, Callback):
                            self.data.append(obj)
