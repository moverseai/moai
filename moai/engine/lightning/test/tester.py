import moai.checkpoint.lightning as mickpt
import moai.log.lightning as milog
from moai.engine.callbacks import PerBatchCallback

import pytorch_lightning
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging

__all__ = ["LightningTester"]

log = logging.getLogger(__name__)


class LightningTester(pytorch_lightning.Trainer):
    def __init__(self,
        logging:                    omegaconf.DictConfig=None,
        callbacks:                  omegaconf.DictConfig=None,
        model_callbacks:            typing.Sequence[pytorch_lightning.Callback]=None,
        default_root_dir:           typing.Optional[str]=None,
        process_position:           int=0,
        num_nodes:                  int=1,
        num_processes:              int=1,
        devices:                    typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        tpu_cores:                  typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        log_gpu_memory:             typing.Optional[str]=None,
        max_steps:                  typing.Optional[int]=-1,
        min_steps:                  typing.Optional[int]=None,
        limit_test_batches:         typing.Union[int, float]=1.0,
        limit_predict_batches:      typing.Union[int, float]=1.0,
        accelerator:                typing.Optional[typing.Union[str, pytorch_lightning.accelerators.Accelerator]]=None,
        strategy:                   typing.Optional[str]=None,
        sync_batchnorm:             bool=False,
        precision:                  int=32,
        enable_model_summary:       bool=True,
        weights_summary:            typing.Optional[str]='full',
        weights_save_path:          typing.Optional[str]=None,        
        resume_from_checkpoint:     typing.Optional[str]=None,
        profiler:                   typing.Optional[typing.Union[pytorch_lightning.profilers.Profiler, bool, str]]=None,
        benchmark:                  bool=False,
        deterministic:              bool=True,
        replace_sampler_ddp:        bool=True,
        prepare_data_per_node:      bool=True,
        plugins:                    typing.Optional[list]=None,
        amp_backend:                str='native',
        amp_level:                  str='O2',
        **kwargs
    ):
        # logger = hyu.instantiate(logging)\
        #     if logging is not None else milog.NoOp()
        loggers = []
        if logging is not None:
            for logger in logging["loggers"].values():
                if logger is not None:
                    loggers.append(hyu.instantiate(logger))
        else:
            log.warning("No loggers defined, using default logger.")
        pytl_callbacks = [hyu.instantiate(c) for c in callbacks.values()]\
            if callbacks is not None else []
        if model_callbacks:
            pytl_callbacks.extend(model_callbacks)
        pytl_callbacks.append(PerBatchCallback())
        super(LightningTester, self).__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=loggers,
            callbacks=pytl_callbacks,
            fast_dev_run=False,
            max_epochs=1,
            min_epochs=1,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=None,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=0.0,
            val_check_interval=1,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            # enable_checkpointing NOT needed in evaluation mode
            enable_progress_bar=True, #TODO: expose as parameter
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=1,
            gradient_clip_val=0.0,
            # gradient_clip_algorithm NOT needed in evaluation mode
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=True, #NOTE: @PT2.0, check if needed
            use_distributed_sampler=True, #NOTE: @PT2.0, check if needed
            profiler=profiler,
            detect_anomaly=False,
            barebones=False, #NOTE: @PT2.0, check if needed
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=False,
            default_root_dir=None if not default_root_dir else default_root_dir,
        )

    def run(self, model):
        return self.test(model, verbose=False)