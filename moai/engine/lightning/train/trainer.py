import moai.log.lightning as milog
from moai.engine.callbacks import PerBatchCallback

import pytorch_lightning
import hydra.utils as hyu
import omegaconf.omegaconf
import typing

__all__ = ["LightningTrainer"]


class LightningTrainer(pytorch_lightning.Trainer):
    def __init__(
        self,
        accelerator: typing.Optional[
            typing.Union[str, pytorch_lightning.accelerators.Accelerator]
        ] = "auto",
        strategy: typing.Optional[
            typing.Union[str, pytorch_lightning.strategies.Strategy]
        ] = "auto",
        devices: typing.Optional[typing.Union[typing.List[int], str, int]] = "auto",
        num_nodes: int = 1,
        precision: int = 32,
        logging: omegaconf.DictConfig = None,
        checkpoint: omegaconf.DictConfig = None,
        regularization: omegaconf.DictConfig = None,
        callbacks: omegaconf.DictConfig = None,
        model_callbacks: typing.Sequence[pytorch_lightning.Callback] = None,
        default_root_dir: typing.Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",  # NOTE: @PTL1.5
        process_position: int = 0,
        num_processes: int = 1,
        gpus: typing.Optional[typing.Union[typing.List[int], str, int]] = None,
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
        auto_select_gpus: bool = False,
        tpu_cores: typing.Optional[typing.Union[typing.List[int], str, int]] = None,
        log_gpu_memory: typing.Optional[str] = None,
        progress_bar_refresh_rate: int = 1,
        overfit_batches: float = 0.0,
        track_grad_norm: int = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: bool = False,
        accumulate_grad_batches: typing.Union[
            int, typing.Dict[int, int], typing.List[list]
        ] = 1,
        max_epochs: int = 1000,
        min_epochs: int = 1,
        max_steps: typing.Optional[int] = -1,
        min_steps: typing.Optional[int] = None,
        limit_train_batches: typing.Union[int, float] = 1.0,
        limit_val_batches: typing.Union[int, float] = 1.0,
        limit_test_batches: typing.Union[int, float] = 1.0,
        limit_predict_batches: typing.Union[int, float] = 1.0,  # NOTE: @PTL1.5
        val_check_interval: typing.Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 10,
        sync_batchnorm: bool = False,
        enable_model_summary: bool = True,
        weights_summary: typing.Optional[str] = "full",
        weights_save_path: typing.Optional[str] = None,
        num_sanity_val_steps: int = 2,
        # NOTE: @PTL1.5 truncated_bptt_steps:       typing.Optional[int]=None,
        resume_from_checkpoint: typing.Optional[str] = None,
        profiler: typing.Optional[
            typing.Union[pytorch_lightning.profilers.Profiler, str]
        ] = None,
        benchmark: bool = False,
        deterministic: bool = True,
        reload_dataloaders_every_epoch: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: typing.Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,  # NOTE: @PTL1.5
        auto_scale_batch_size: typing.Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: typing.Optional[list] = None,
        amp_backend: str = "native",
        amp_level: str = "O2",
        move_metrics_to_cpu: bool = False,  # NOTE: @PTL1.5
        terminate_on_nan: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        stochastic_weight_avg: bool = False,
        # NOTE: @PTL1.5 distributed_backend:        typing.Optional[str]=None,
        # NOTE: @PTL1.5 automatic_optimization:     bool=True,
        **kwargs
    ):
        # if logging and '_target_' not in logging: #TODO: needs a workaround for other viz types (e.g. not visdom) if they are moved to top level
        #     #TODO: wrap config field setting into a helper method
        #     omegaconf.OmegaConf.set_struct(logging, False)
        #     logging['_target_'] = 'moai.log.lightning.Collection'
        #     omegaconf.OmegaConf.set_struct(logging, True)
        # logger = hyu.instantiate(logging)\
        #     if logging is not None else milog.NoOp()
        pytl_callbacks = (
            [hyu.instantiate(c) for c in callbacks.values()]
            if callbacks is not None
            else []
        )
        pytl_callbacks.append(PerBatchCallback())
        checkpoint_callback = False
        if checkpoint is not None:
            pytl_callbacks.append(hyu.instantiate(checkpoint))
            checkpoint_callback = True
        if regularization is not None:
            pytl_callbacks.append(hyu.instantiate(regularization))
        if model_callbacks:
            pytl_callbacks.extend(model_callbacks)
        super(LightningTrainer, self).__init__(
            accelerator=accelerator,  # Union[str, Accelerator] = "auto",
            strategy=strategy,  # Union[str, Strategy] = "auto",
            devices=gpus,  # Union[List[int], str, int] = "auto",
            num_nodes=num_nodes,  # int = 1,
            precision=precision,
            logger=[hyu.instantiate(logger) for logger in logging["loggers"].values()],
            callbacks=pytl_callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=None,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=checkpoint_callback,  # PL 2.0
            # enable_progress_bar: Optional[bool] = None, # PL 2.0
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            # inference_mode: bool = True, # PL 2.0
            # use_distributed_sampler: bool = True, # PL 2.0
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            # barebones: bool = False, # PL 2.0
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=None if not default_root_dir else default_root_dir,
        )

    def run(self, model):
        self.fit(model)
        # self.test(model)
