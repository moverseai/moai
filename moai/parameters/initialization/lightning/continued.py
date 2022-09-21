from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.upgrade_checkpoint import KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from apex import amp
except ImportError:
    amp = None

import typing
import torch
import logging

log = logging.getLogger(__name__)

__all__ = ["Continued"]

class Continued(Callback):
    def __init__(self,
        filename:           str,
    ):
        super(Continued, self).__init__()
        self.filename = filename
    
    def on_fit_start(self, trainer, pl_module):        
        checkpoint = pl_load(self.filename,
            map_location=lambda storage, loc: storage
        )
        if 'optimizer_states' not in checkpoint or 'lr_schedulers' not in checkpoint:
            error = 'Trying to restore training state but checkpoint contains only the model.'
            ' This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.'
            # raise KeyError(error)
            log.error(error)

        if any([key in checkpoint for key in DEPRECATED_CHECKPOINT_KEYS]):
            error = "The checkpoint you're attempting to load follows an"
            " outdated schema. You can upgrade to the current schema by running"
            " `python -m pytorch_lightning.utilities.upgrade_checkpoint --file model.ckpt`"
            " where `model.ckpt` is your checkpoint file."
            log.error(error)            
            raise ValueError(error)

        # restore amp scaling
        if trainer.amp_backend == AMPType.NATIVE and 'native_amp_scaling_state' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['native_amp_scaling_state'])
        elif trainer.amp_backend == AMPType.APEX and 'amp_scaling_state' in checkpoint:
            amp.load_state_dict(checkpoint['amp_scaling_state'])

        trainer.model.load_state_dict(checkpoint['state_dict'], strict=True)
        trainer.fit_loop.global_step = checkpoint['global_step']
        trainer.fit_loop.current_epoch = checkpoint['epoch']
        #trainer.global_step = checkpoint['global_step']
        #trainer.current_epoch = checkpoint['epoch']

        # crash if max_epochs is lower then the current epoch from the checkpoint
        if trainer.current_epoch > trainer.max_epochs:
            m = f"""
            you restored a checkpoint with current_epoch={trainer.current_epoch}
            but the Trainer(max_epochs={trainer.max_epochs})
            """
            # raise MisconfigurationException(m)
            log.warning(m)

        # Division deals with global step stepping once per accumulated batch
        # Inequality deals with different global step for odd vs even num_training_batches
        n_accum = 1 if trainer.accumulate_grad_batches is None else trainer.accumulate_grad_batches
        expected_steps = trainer.num_training_batches / n_accum
        if trainer.num_training_batches != 0 and trainer.global_step % expected_steps > 1:
            msg = "You're resuming from a checkpoint that ended mid-epoch. "
            "This can cause unreliable results if further training is done, "
            "consider using an end of epoch checkpoint. "
            log.warning(msg)

        log.info(f"Loading optimizer state from {self.filename} @ epoch {trainer.current_epoch} & step {trainer.global_step}.")
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(trainer.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

            # move optimizer to GPU 1 weight at a time
            # avoids OOM
            if trainer.root_gpu is not None:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(trainer.root_gpu)
            optimizer.param_groups[0]['capturable'] = True #TODO: bug, https://github.com/pytorch/pytorch/issues/80809


        # restore the lr schedulers
        lr_schedulers = checkpoint['lr_schedulers']
        for scheduler, lrs_state in zip(trainer.lr_schedulers, lr_schedulers):
            scheduler['scheduler'].load_state_dict(lrs_state)