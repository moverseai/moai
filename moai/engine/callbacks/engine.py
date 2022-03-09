from collections import UserList
from pytorch_lightning import Callback

import  moai.utils.engine as mieng
import logging

log = logging.getLogger(__name__)

class EngineCallbacks(UserList):
    def __init__(self,
        **kwargs
    ):
        super().__init__()
        for k, v in kwargs.items():
            if isinstance(v, Callback):
                self.data.append(v)
            if isinstance(v, mieng.Collection):
                for i in v.items():
                    if isinstance(i, Callback):
                        self.data.append(i)