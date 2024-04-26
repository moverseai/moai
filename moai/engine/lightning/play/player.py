# import moai.log.lightning as milog
from moai.engine.callbacks.per_batch import PerBatchCallback
import pytorch_lightning
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["LightningPlayer"]

class LightningPlayer(pytorch_lightning.Trainer):
    def __init__(self,
        engine_callbacks:       typing.Sequence[pytorch_lightning.Callback]=[],
        logging:                omegaconf.DictConfig=None,
        num_nodes:              int=1,
        devices:                typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        tpu_cores:              typing.Optional[int]=None,
        overfit_batches:        float=0.0,
        fast_dev_run:           bool=False,
        limit_train_batches:    float=1.0,
        limit_val_batches:      float=1.0,
        limit_test_batches:     float=1.0,
        val_check_interval:     float=1.0,
        #distributed_backend: typing.Optional[str]=None, NOTE: @PLT1.5
        weights_summary:        typing.Optional[str]='full',
        amp_level:              str='O2',
        accelerator:            typing.Optional[
            typing.Union[str, pytorch_lightning.accelerators.Accelerator]
        ]='auto',
        **kwargs
    ):
        # logger = hyu.instantiate(logging) if logging is not None else milog.NoOp()
        loggers = []
        if logging is not None:
            for logger in logging["loggers"].values():
                if logger is not None:
                    loggers.append(hyu.instantiate(logger))
        else:
            log.warning("No loggers defined, using default logger.")
        engine_callbacks.append(PerBatchCallback())
        super(LightningPlayer, self).__init__(
            devices=devices,
            accelerator=accelerator,
            logger=loggers,
            # logger=logger,
            callbacks=engine_callbacks,
            max_epochs=1,
            min_epochs=1,
            fast_dev_run=fast_dev_run,
            num_nodes=num_nodes,
            overfit_batches=overfit_batches,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            val_check_interval=val_check_interval,
            #distributed_backend=distributed_backend,
            # weights_summary=weights_summary,
            # amp_level=amp_level,
            **kwargs
        )

    def play(self, model):
        self.fit(model)
        log.info("Train & validation data completed, continuing with test data.")
        self.test(model)