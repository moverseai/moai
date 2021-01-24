import moai.utils.engine as mieng

import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Single"]

class Single(mieng.Collection):
    def __init__(self,
        items: omegaconf.DictConfig,
        arguments: typing.Sequence[typing.Any]=None,
        name: str="items",
    ):
        super(Single, self).__init__(
            items=omegaconf.DictConfig(
                {next(iter(items.keys())): next(iter(items.values()))}
            ), name=name,
            arguments=[next(iter(arguments))] if arguments else None
        )
        if len(items.values()) > 1:
            log.warning("Multiple items have been defined,"
                "but a single item scheme is used.")