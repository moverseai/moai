import moai.checkpoint.lightning as mickpt
import moai.log.lightning as milog

import pytorch_lightning
import hydra.utils as hyu
import omegaconf.omegaconf
import typing

__all__ = ["LightningTester"]

class LightningTester(pytorch_lightning.Trainer):
    def __init__(self,
        logging:                    omegaconf.DictConfig=None,
        callbacks:                  omegaconf.DictConfig=None,
        default_root_dir:           typing.Optional[str]=None,
        process_position:           int=0,
        num_nodes:                  int=1,
        num_processes:              int=1,
        gpus:                       typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        tpu_cores:                  typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        log_gpu_memory:             typing.Optional[str]=None,
        max_steps:                  typing.Optional[int]=None,
        min_steps:                  typing.Optional[int]=None,
        limit_test_batches:         typing.Union[int, float]=1.0,
        accelerator:                typing.Optional[typing.Union[str, pytorch_lightning.accelerators.Accelerator]]=None,
        sync_batchnorm:             bool=False,
        precision:                  int=32,
        weights_summary:            typing.Optional[str]='full',
        weights_save_path:          typing.Optional[str]=None,
        truncated_bptt_steps:       typing.Optional[int]=None,
        resume_from_checkpoint:     typing.Optional[str]=None,
        profiler:                   typing.Optional[typing.Union[pytorch_lightning.profiler.BaseProfiler, bool, str]]=None,
        benchmark:                  bool=False,
        deterministic:              bool=True,
        replace_sampler_ddp:        bool=True,
        prepare_data_per_node:      bool=True,
        plugins:                    typing.Optional[list]=None,
        amp_backend:                str='native',
        amp_level:                  str='O2',
        distributed_backend:        typing.Optional[str]=None,
        **kwargs
    ):
        logger = hyu.instantiate(logging)\
            if logging is not None else milog.NoOp()
        pytl_callbacks = [hyu.instantiate(c) for c in callbacks.values()]\
            if callbacks is not None else []                
        super(LightningTester, self).__init__(
            logger=logger,
            checkpoint_callback=False,
            callbacks=pytl_callbacks,
            default_root_dir=None if not default_root_dir else default_root_dir,
            gradient_clip_val=0.0,
            process_position=process_position,
            num_nodes=num_nodes,
            gpus=gpus,
            auto_select_gpus=False,
            tpu_cores=tpu_cores,
            log_gpu_memory=log_gpu_memory,
            progress_bar_refresh_rate=1,
            overfit_batches=0.0,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            accumulate_grad_batches=1,
            max_epochs=1,
            min_epochs=1,
            max_steps=max_steps,
            min_steps=min_steps,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=limit_test_batches,
            val_check_interval=1,
            flush_logs_every_n_steps=1,
            log_every_n_steps=1,
            accelerator=accelerator,
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            weights_summary=weights_summary,
            weights_save_path=weights_save_path,
            num_sanity_val_steps=0,
            truncated_bptt_steps=truncated_bptt_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            reload_dataloaders_every_epoch=False,
            auto_lr_find=False,
            replace_sampler_ddp=replace_sampler_ddp,
            terminate_on_nan=False,
            auto_scale_batch_size=False,
            prepare_data_per_node=prepare_data_per_node,
            plugins=plugins,
            amp_backend=amp_backend,            
            distributed_backend=distributed_backend,
            amp_level=amp_level,
            automatic_optimization=False,
            **kwargs
        )

    def run(self, model):
        return self.test(model, verbose=False)