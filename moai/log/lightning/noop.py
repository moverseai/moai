import pytorch_lightning
import typing
import omegaconf.omegaconf

__all__ = ["NoOp"]

class NoOp(pytorch_lightning.loggers.base.LightningLoggerBase):
    def __init__(self,
        loggers: omegaconf.DictConfig=None,
        **kwargs
    ):
        super(NoOp, self).__init__()

    @property
    def name(self):
        return "NoLogging"
        
    @pytorch_lightning.loggers.base.rank_zero_only
    def log_metrics(self, 
        metrics: typing.Dict[str, typing.Any],
        step: int
    ):
       pass

    @pytorch_lightning.loggers.base.rank_zero_only
    def log_hyperparams(self,
        params: typing.Dict[str, typing.Any], #TODO: or namespace object ?
    ):
        pass

    @pytorch_lightning.loggers.base.rank_zero_only
    def save(self):
        pass

    @pytorch_lightning.loggers.base.rank_zero_only
    def finalize(self, 
        status: str
    ):
        pass

    @property
    def rank(self):
        return 0

    @rank.setter
    def rank(self, value: int):
        pass

    @property
    def version(self):
        return 0

    @property
    def experiment(self) -> typing.Any:
        return self.name