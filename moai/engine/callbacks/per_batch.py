import torch
import pytorch_lightning
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['PerBatchCallback']


class PerBatchCallback(pytorch_lightning.Callback):
    def __init__(self):
        super(PerBatchCallback, self).__init__()
        log.info('Instantiated PerBatchCallback')

    def on_train_batch_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
        outputs: typing.Dict[
            str,
            typing.Union[
                torch.Tensor,
                typing.Sequence[torch.Tensor],
                typing.Dict[str, torch.Tensor],
            ],
        ],
        batch: typing.Dict[
            str,
            typing.Union[
                torch.Tensor,
                typing.Sequence[torch.Tensor],
                typing.Dict[str, torch.Tensor],
            ],
        ],
        batch_idx: int,
        unused: typing.Optional[int] = 0,
    ) -> None:
        # replace training_step_end used in PTL 1.5
        if trainer.global_step and (trainer.global_step % pl_module.visualization.interval == 0):
            pl_module.visualization(outputs['tensors'], trainer.global_step)
        if trainer.global_step and (trainer.global_step % pl_module.exporter.interval == 0):
            pl_module.exporter(outputs['tensors'], trainer.global_step)
        return outputs['loss']