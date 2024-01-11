# instantiate hydra logging for torchserve
import omegaconf.omegaconf
import logging



class HydraLogging(object):
    def __init__(
        self,
        # hyu: typing.Dict = None,
        job: omegaconf.DictConfig = None,
    ):
        # NOTE: this is a workaround for the fact that torchserve
        # does not support hydra logging
        # Applying the logging configuration
        logging.config.dictConfig(omegaconf.OmegaConf.to_container(job, resolve=True))
