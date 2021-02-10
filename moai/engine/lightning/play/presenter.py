from moai.data.iterator import Indexed

import moai.networks.lightning as minet

import torch
import pytorch_lightning 
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Presenter"]

class Presenter(minet.FeedForward):
    def __init__(self, 
        feedforward:        omegaconf.DictConfig=None,
        monads:             omegaconf.DictConfig=None,
        visualization:      omegaconf.DictConfig=None,
        data:               omegaconf.DictConfig=None,
    ):        
        super(Presenter, self).__init__(
            feedforward=feedforward, data=data,
            monads=monads, visualization=visualization,            
        )
        self.param = torch.nn.Linear(1, 1) # dummy layer
        self.global_test_step = 0

    def forward(self,
        x: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        return x

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
        optimizer_idx:          int=None,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        postprocessed = self.postprocess(prediction)
        total_loss = torch.zeros(1, requires_grad=True)
        return { 'loss': total_loss, 'tensors': postprocessed }

    def training_step_end(self,
        train_outputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]
    ) -> None:
        if self.global_step and (self.global_step % self.visualizer.interval == 0):
            self.visualizer(train_outputs['tensors'])
        return train_outputs['loss']

    def validation_step(self,
        batch: typing.Dict[str, torch.Tensor],
        batch_nb: int
    ) -> dict:
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        outputs = self.postprocess(prediction)
        return None

    def validation_epoch_end(self,
        outputs: typing.List[dict]
    ) -> dict:
        pass
    
    def test_step(self, 
        batch: typing.Dict[str, torch.Tensor],
        batch_nb: int
    ) -> dict:
        preprocessed = self.preprocess(batch)        
        prediction = self(preprocessed)
        outputs = self.postprocess(prediction)
        self.global_test_step += 1
        return torch.zeros(1), outputs

    def test_step_end(self,
        metrics_tensors: typing.Tuple[typing.Dict[str, torch.Tensor], typing.Dict[str, torch.Tensor]],        
    ) -> None:
        metrics, tensors = metrics_tensors
        if self.global_test_step and (self.global_test_step % self.exporter.interval == 0):
            self.exporter(tensors)
        if self.global_test_step and (self.global_test_step % self.visualizer.interval == 0):
            self.visualizer(tensors)
        return metrics

    def test_epoch_end(self, 
        outputs: typing.List[dict]
    ) -> dict:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.train.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.train.iterator._target_.split('.')[-1]}) train set data iterator")
            train_iterator = hyu.instantiate(self.data.train.iterator)
        else:
            train_iterator = Indexed(
                self.data.train.iterator.datasets,
                self.data.train.iterator.augmentation if hasattr(self.data.train.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.train, 'loader'):
            log.error("Train data loader missing. Please add a data loader (i.e. \'- data/train/loader: torch\') entry in the configuration.")
        else:
            train_loader = hyu.instantiate(self.data.train.loader, train_iterator)
        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.val.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.val.iterator._target_.split('.')[-1]}) validation set data iterator")
            val_iterator = hyu.instantiate(self.data.val.iterator)
        else:
            val_iterator = Indexed(
                self.data.val.iterator.datasets,
                self.data.val.iterator.augmentation if hasattr(self.data.val.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.val, 'loader'):
            log.error("Validation data loader missing. Please add a data loader (i.e. \'- data/val/loader: torch\') entry in the configuration.")
        else:
            validation_loader = hyu.instantiate(self.data.val.loader, val_iterator)
        return validation_loader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.test.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.test.iterator._target_.split('.')[-1]}) test set data iterator")
            test_iterator = hyu.instantiate(self.data.test.iterator)
        else:
            test_iterator = Indexed(
                self.data.test.iterator.datasets,
                self.data.test.iterator.augmentation if hasattr(self.data.test.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.test, 'loader'):
            log.error("Test data loader missing. Please add a data loader (i.e. \'- data/test/loader: torch\') entry in the configuration.")
        else:
            test_loader = hyu.instantiate(self.data.test.loader, test_iterator)
        return test_loader