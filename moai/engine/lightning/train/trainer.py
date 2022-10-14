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
        loop:                       omegaconf.DictConfig=None,
        model_callbacks:            typing.Sequence[pytorch_lightning.Callback]=None,
        default_root_dir:           typing.Optional[str]=None,
        gradient_clip_val:          float=0.0,
        gradient_clip_algorithm:    str='norm', #NOTE: @PTL1.5    
        process_position:           int=0,
        num_nodes:                  int=1,
        num_processes:              int=1,
        devices:                    typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        gpus:                       typing.Optional[typing.Union[typing.List[int], str, int]]=None,
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
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
        limit_predict_batches:      typing.Union[int, float]=1.0, #NOTE: @PTL1.5
        val_check_interval:         typing.Union[int, float]=1.0,
        flush_logs_every_n_steps:   int=100,
        log_every_n_steps:          int=10,
        accelerator:                typing.Optional[typing.Union[str, pytorch_lightning.accelerators.Accelerator]]=None,
        strategy:                   str=None,
        sync_batchnorm:             bool=False,
        precision:                  int=32,
        enable_model_summary:       bool=True,
        weights_summary:            typing.Optional[str]='full',
        weights_save_path:          typing.Optional[str]=None,
        num_sanity_val_steps:       int=2,
         #NOTE: @PTL1.5 truncated_bptt_steps:       typing.Optional[int]=None,
        resume_from_checkpoint:     typing.Optional[str]=None,
        profiler:                   typing.Optional[typing.Union[pytorch_lightning.profiler.BaseProfiler, bool, str]]=None,
        benchmark:                  bool=False,
        deterministic:              bool=True,
        reload_dataloaders_every_epoch: bool=False,
        reload_dataloaders_every_n_epochs: int=0,
        auto_lr_find:               typing.Union[bool, str]=False,        
        replace_sampler_ddp:        bool=True,
        detect_anomaly:             bool=False, #NOTE: @PTL1.5
        auto_scale_batch_size:      typing.Union[str, bool]=False,
        prepare_data_per_node:      bool=True,
        plugins:                    typing.Optional[list]=None,
        amp_backend:                str='native',
        amp_level:                  str='O2',
        move_metrics_to_cpu:        bool=False, #NOTE: @PTL1.5
        terminate_on_nan:           bool=False,
        multiple_trainloader_mode:  str="max_size_cycle",
        stochastic_weight_avg:      bool=False,                        
        #NOTE: @PTL1.5 distributed_backend:        typing.Optional[str]=None,
        #NOTE: @PTL1.5 automatic_optimization:     bool=True,
        **kwargs
    ):
        if logging and '_target_' not in logging: #TODO: needs a workaround for other viz types (e.g. not visdom) if they are moved to top level
            #TODO: wrap config field setting into a helper method
            omegaconf.OmegaConf.set_struct(logging, False)
            logging['_target_'] = 'moai.log.lightning.Collection'
            omegaconf.OmegaConf.set_struct(logging, True)
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
        if model_callbacks:
            pytl_callbacks.extend(model_callbacks)
        super(LightningTrainer, self).__init__(
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=pytl_callbacks,
            default_root_dir=None if not default_root_dir else default_root_dir,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm, #NOTE: @PTL1.5
            process_position=process_position,
            num_nodes=num_nodes,
            devices=devices, #NOTE @PTL1.5 #TODO: check if needed
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
            max_time=None, #NOTE @PTL1.5 #TODO: check if needed
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches, #NOTE: @PTL1.5
            val_check_interval=val_check_interval,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            strategy=strategy, #NOTE: @PTL1.5
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            enable_model_summary=enable_model_summary,
            weights_summary=weights_summary,
            weights_save_path=weights_save_path,
            num_sanity_val_steps=num_sanity_val_steps,
             #NOTE: @PTL1.5 truncated_bptt_steps=truncated_bptt_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            reload_dataloaders_every_epoch=reload_dataloaders_every_epoch,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            auto_lr_find=auto_lr_find,
            detect_anomaly=detect_anomaly, #NOTE: @PTL1.5
            replace_sampler_ddp=replace_sampler_ddp,
            terminate_on_nan=terminate_on_nan,
            auto_scale_batch_size=auto_scale_batch_size,
            prepare_data_per_node=prepare_data_per_node,
            plugins=plugins,
            amp_backend=amp_backend,            
            #NOTE: @PTL1.5 fixdistributed_backend=distributed_backend,
            amp_level=amp_level,
            #NOTE: @PTL1.5 fixautomatic_optimization=automatic_optimization,
            move_metrics_to_cpu=move_metrics_to_cpu,
            multiple_trainloader_mode=multiple_trainloader_mode,
            stochastic_weight_avg=stochastic_weight_avg,
            num_processes=num_processes, #NOTE: @PTL1.5 fix
            **kwargs
        )
        if loop is not None:
            from pytorch_lightning.loops import EvaluationLoop, PredictionLoop, TrainingBatchLoop, TrainingEpochLoop
            fit_loop = hyu.instantiate(loop)
            # fit_loop = FitLoop(
            #     min_epochs=(1 if (min_epochs is None and min_steps is None and max_time is None) else min_epochs),
            #     max_epochs=(
            #         max_epochs if max_epochs is not None else (1000 if (max_steps == -1 and max_time is None) else -1)
            #     ),
            # )
            training_epoch_loop = TrainingEpochLoop(min_steps, max_steps)
            training_batch_loop = TrainingBatchLoop()
            training_validation_loop = EvaluationLoop()
            training_epoch_loop.connect(batch_loop=training_batch_loop, val_loop=training_validation_loop)
            fit_loop.connect(epoch_loop=training_epoch_loop)
            self.fit_loop = fit_loop
            


    def run(self, model):
        self.fit(model)