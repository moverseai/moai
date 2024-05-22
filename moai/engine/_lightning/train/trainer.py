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
        ] = "auto", # 
        strategy: typing.Optional[
            typing.Union[str, pytorch_lightning.strategies.Strategy]
        ] = "auto",
        devices: typing.Optional[typing.Union[typing.List[int], str, int]] = "auto",
        gpus: typing.Optional[typing.Union[typing.List[int], str, int]] = None,  #TODO: remove this and use devices instead
        num_nodes: int = 1,
        precision: int = 32,
        logging: omegaconf.DictConfig = None,
        checkpoint: omegaconf.DictConfig = None,
        regularization: omegaconf.DictConfig = None,
        callbacks: omegaconf.DictConfig = None,
        model_callbacks: typing.Sequence[pytorch_lightning.Callback] = None,
        default_root_dir: typing.Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",
        # process_position: int = 0, # NOTE: removed in @PTL2.1
        # num_processes: int = 1, # NOTE: removed in @PTL2.1
        # gpus: typing.Optional[typing.Union[typing.List[int], str, int]] = None, # NOTE: removed in @PTL2.1
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
        # auto_select_gpus: bool = False, # NOTE: removed in @PTL2.1
        # tpu_cores: typing.Optional[typing.Union[typing.List[int], str, int]] = None, # NOTE: removed in @PTL2.1
        # log_gpu_memory: typing.Optional[str] = None, # NOTE: removed in @PTL2.1
        # progress_bar_refresh_rate: int = 1, # NOTE: removed in @PTL2.1
        overfit_batches: float = 0.0,
        # track_grad_norm: int = -1, # NOTE: removed in @PTL2.1
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
        # flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 10,
        sync_batchnorm: bool = False,
        enable_model_summary: bool = True,
        # weights_summary: typing.Optional[str] = "full", # NOTE: removed in @PTL2.1
        # weights_save_path: typing.Optional[str] = None, # NOTE: removed in @PTL2.1
        num_sanity_val_steps: int = 2,
        # NOTE: @PTL1.5 truncated_bptt_steps:       typing.Optional[int]=None,
        # resume_from_checkpoint: typing.Optional[str] = None,
        profiler: typing.Optional[
            typing.Union[pytorch_lightning.profilers.Profiler, str]
        ] = None,
        benchmark: bool = False,
        deterministic: bool = True,
        # reload_dataloaders_every_epoch: bool = False, # NOTE: removed in @PTL2.1
        reload_dataloaders_every_n_epochs: int = 0,
        # auto_lr_find: typing.Union[bool, str] = False, NOTE:  removed in @PTL2.1
        # replace_sampler_ddp: bool = True, # NOTE: removed in @PTL2.1
        detect_anomaly: bool = False,
        # auto_scale_batch_size: typing.Union[str, bool] = False,
        # prepare_data_per_node: bool = True, # NOTE: removed in @PTL2.1
        plugins: typing.Optional[list] = None,
        # amp_backend: str = "native", # NOTE: removed in @PTL2.1
        # amp_level: str = "O2", # NOTE: removed in @PTL2.1
        # terminate_on_nan: bool = False, # NOTE: removed in @PTL2.1
        # multiple_trainloader_mode: str = "max_size_cycle", # NOTE: removed in @PTL2.1
        # stochastic_weight_avg: bool = False, # NOTE: removed in @PTL2.1
        # NOTE: @PTL1.5 distributed_backend:        typing.Optional[str]=None,
        # NOTE: @PTL1.5 automatic_optimization:     bool=True,
        **kwargs
    ):
        r"""Wraps PyTorch Lightning Trainer with additional features.

        Args:
            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
                as well as custom accelerator instances.

            strategy: Strategy controls the model distribution across training, evaluation, and prediction to 
            be used by the Trainer. It can be controlled by passing different strategy with aliases ("ddp", "ddp_spawn", "deepspeed" and so on) as well as a custom strategy to the strategy parameter for Trainer.

            devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
                (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
                automatic selection based on the chosen accelerator. Default: ``"auto"``.
            
            num_nodes: Number of GPU nodes for distributed training.
                Default: ``1``.

            precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
                16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
                Can be used on CPU, GPU, TPUs, HPUs or IPUs.
                Default: ``'32-true'``.
            
            logging: OmegaConf DictConfig for logging configuration.

            checkpoint: OmegaConf DictConfig for checkpoint configuration.

            regularization: OmegaConf DictConfig for regularization configuration.
            
            callbacks: OmegaConf DictConfig for callbacks configuration.

            model_callbacks: Sequence of PyTorch Lightning Callbacks to be added to the model.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``None``.
            
            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
                Default: ``None``.
            
            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.
            
            overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
                Default: ``0.0``.
            
            
            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches. An ``int`` value can only be higher than the number of training batches when
                ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
                across epochs or during.
                Default: ``1.0``. 
            
            check_val_every_n_epoch: Perform a validation loop every after every `N` training epochs. If ``None``,
                validation will be done solely based on the number of training batches, requiring ``val_check_interval``
                to be an integer value.
                Default: ``1``.
            
            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.
            
            accumulate_grad_batches: Accumulates gradients over k batches before stepping the optimizer.
                Default: 1.
            
            profiler: To profile individual steps during training and assist in identifying bottlenecks.
                Default: ``None``.
            
            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.
            
            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.
                Default: ``2``.
            
            benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
                The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
                (``False`` if not manually set). If :paramref:`~pytorch_lightning.trainer.trainer.Trainer.deterministic`
                is set to ``True``, this will default to ``False``. Override to manually set a different value.
                Default: ``None``.
            
            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
                Default: ``None``.
            
        
        """
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
        
        loggers = []
        if logging is not None:
            for logger in logging["loggers"].values():
                if logger is not None:
                    loggers.append(hyu.instantiate(logger))

        super(LightningTrainer, self).__init__(
            accelerator=accelerator,  # Union[str, Accelerator] = "auto",
            strategy=strategy,  # Union[str, Strategy] = "auto",
            # devices=devices,  # Union[List[int], str, int] = "auto",
            devices=gpus, # TODO: remove this and use devices instead
            num_nodes=num_nodes,  # int = 1,
            precision=precision,
            logger=loggers,
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

    def run(self, model, resume_from_checkpoint: str = None):
        self.fit(model, ckpt_path=resume_from_checkpoint)
        # self.test(model)
