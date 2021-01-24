from collections.abc import Callable

import torch
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Interval"]

class Interval(Callable):
    def __init__(self,
        batch_interval:int
    ):
        self.batch_interval = batch_interval

    @property
    def interval(self):
        return self.batch_interval