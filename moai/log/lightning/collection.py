import pytorch_lightning
import omegaconf.omegaconf
import hydra.utils as hyu
import logging
import functools

log = logging.getLogger(__name__)

__all__ = ["Loggers"]

class Loggers(pytorch_lightning.loggers.LoggerCollection):
    def __init__(self,
        loggers: omegaconf.DictConfig={},
        name: str="Sequential",
        version: int=0
    ):
        super(Loggers, self).__init__(logger_iterable=[
            hyu.instantiate(logger) for logger in loggers.values()
        ])
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        names = [logger.name for logger in self]
        return functools.reduce(lambda seed, item: f"{seed}_{item}", set(names))