import moai.checkpoint.lightning as mickpt
import moai.log.lightning as milog

import pytorch_lightning
import hydra.utils as hyu
import omegaconf.omegaconf
import typing

__all__ = ["LightningTrainer"]

class LightningTrainer(pytorch_lightning.Trainer):
    def __init__(self,
        logging:                    omegaconf.DictConfig=None,
        checkpoint:                 omegaconf.DictConfig=None,
        regularization:             omegaconf.DictConfig=None,
        callbacks:                  omegaconf.DictConfig=None,
        default_root_dir:           typing.Optional[str]=None,
        gradient_clip_val:          float=0.0,
        process_position:           int=0,
        num_nodes:                  int=1,
        num_processes:              int=1,
        gpus:                       typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        auto_select_gpus:           bool=False,
        tpu_cores:                  typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        log_gpu_memory:             typing.Optional[str]=None,
        progress_bar_refresh_rate:  int=1,
        overfit_batches:            float=0.0,
        track_grad_norm:            int=-1,
        check_val_every_n_epoch:    int=1,
        fast_dev_run:               bool=False,
        accumulate_grad_batches:    typing.Union[int, typing.Dict[int, int], typing.List[list]]=1,
        max_epochs:                 int=1000,
        min_epochs:                 int=1,
        max_steps:                  typing.Optional[int]=None,
        min_steps:                  typing.Optional[int]=None,
        limit_train_batches:        typing.Union[int, float]=1.0,
        limit_val_batches:          typing.Union[int, float]=1.0,
        limit_test_batches:         typing.Union[int, float]=1.0,
        val_check_interval:         typing.Union[int, float]=1.0,
        flush_logs_every_n_steps:   int=100,
        log_every_n_steps:          int=10,
        accelerator:                typing.Optional[typing.Union[str, pytorch_lightning.accelerators.Accelerator]]=None,
        sync_batchnorm:             bool=False,
        precision:                  int=32,
        weights_summary:            typing.Optional[str]='full',
        weights_save_path:          typing.Optional[str]=None,
        num_sanity_val_steps:       int=2,
        truncated_bptt_steps:       typing.Optional[int]=None,
        resume_from_checkpoint:     typing.Optional[str]=None,
        profiler:                   typing.Optional[typing.Union[pytorch_lightning.profiler.BaseProfiler, bool, str]]=None,
        benchmark:                  bool=False,
        deterministic:              bool=True,
        reload_dataloaders_every_epoch: bool=False,
        auto_lr_find:               typing.Union[bool, str]=False,
        replace_sampler_ddp:        bool=True,
        terminate_on_nan:           bool=False,
        auto_scale_batch_size:      typing.Union[str, bool]=False,
        prepare_data_per_node:      bool=True,
        plugins:                    typing.Optional[list]=None,
        amp_backend:                str='native',
        amp_level:                  str='O2',
        distributed_backend:        typing.Optional[str]=None,
        automatic_optimization:     bool=True,
        **kwargs
    ):
        logger = hyu.instantiate(logging)\
            if logging is not None else milog.NoOp()
        pytl_callbacks = [hyu.instantiate(c) for c in callbacks.values()]\
            if callbacks is not None else []
        checkpoint_callback = False
        if checkpoint is not None:
            pytl_callbacks.append(hyu.instantiate(checkpoint))
            checkpoint_callback = True
        if regularization is not None:
            pytl_callbacks.append(hyu.instantiate(regularization))
        super(LightningTrainer, self).__init__(
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=pytl_callbacks,
            default_root_dir=None if not default_root_dir else default_root_dir,
            gradient_clip_val=gradient_clip_val,
            process_position=process_position,
            num_nodes=num_nodes,
            gpus=gpus,
            auto_select_gpus=auto_select_gpus,
            tpu_cores=tpu_cores,
            log_gpu_memory=log_gpu_memory,
            progress_bar_refresh_rate=progress_bar_refresh_rate,
            overfit_batches=overfit_batches,
            track_grad_norm=track_grad_norm,
            check_val_every_n_epoch=check_val_every_n_epoch,
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            val_check_interval=val_check_interval,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            weights_summary=weights_summary,
            weights_save_path=weights_save_path,
            num_sanity_val_steps=num_sanity_val_steps,
            truncated_bptt_steps=truncated_bptt_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            reload_dataloaders_every_epoch=reload_dataloaders_every_epoch,
            auto_lr_find=auto_lr_find,
            replace_sampler_ddp=replace_sampler_ddp,
            terminate_on_nan=terminate_on_nan,
            auto_scale_batch_size=auto_scale_batch_size,
            prepare_data_per_node=prepare_data_per_node,
            plugins=plugins,
            amp_backend=amp_backend,            
            distributed_backend=distributed_backend,
            amp_level=amp_level,
            automatic_optimization=automatic_optimization,
            **kwargs
        )

    def run(self, model):
        self.fit(model)