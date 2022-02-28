from collections.abc import Callable

import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import logging
import typing

log = logging.getLogger(__name__)

__all__ = ["Collection"]

class Collection(Callable): #TODO: inherit from UserList as well, need to update items to iteration
    def __init__(self,
        items: omegaconf.DictConfig,
        arguments: typing.Sequence[typing.Any]=None,
        name: str="items",
    ):
        self.name = name
        items = [hyu.instantiate(item) for item in items.values()]\
            if not arguments else [
                hyu.instantiate(item, arg) for item, arg in zip(items.values(), arguments)
            ]
        setattr(self, name, items)
        if arguments and len(items) != len(arguments):
            log.warning(f"Inconsistent item ({len(items)}) and argument ({len(arguments)}) count, the matching subset is only used.")

    def items(self) -> typing.Iterable[typing.Any]:
        return getattr(self, self.name)

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:        
        for item in self.items():
            item(tensors)