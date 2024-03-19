import pytorch_lightning
import typing
import omegaconf.omegaconf

__all__ = ["NoOp"]

class NoOp(pytorch_lightning.loggers.Logger):
    def __init__(self,
        loggers: omegaconf.DictConfig=None,
        **kwargs
    ):
        super(NoOp, self).__init__()

    @property
    def name(self):
        return "NoLogging"
        
    @pytorch_lightning.loggers.logger.rank_zero_experiment
    def log_metrics(self, 
        metrics: typing.Dict[str, typing.Any],
        step: int
    ):
       pass

    def log_hyperparams(self,
        params: typing.Dict[str, typing.Any], #TODO: or namespace object ?
    ):
        pass

    @pytorch_lightning.loggers.logger.rank_zero_experiment
    def save(self):
        pass

    @pytorch_lightning.loggers.logger.rank_zero_experiment
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