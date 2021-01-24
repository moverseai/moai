import moai.utils.engine as mieng

import sys
import torch
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["NoOp"]

class NoOp(mieng.Interval):
    def __init__(self, 
        *args:      typing.Sequence[typing.Any],
        **kwargs:   typing.Mapping[str, typing.Any]
    ):
        super(NoOp, self).__init__(sys.maxsize * 2 + 1) # big enough interval

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        pass