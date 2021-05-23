import pytorch_lightning
from moai.validation.collection import Metrics
from moai.validation.metrics.common.classified import (
    TruePositive,
    TrueNegative,
    FalsePositive,
    FalseNegative,
)
from pytorch_lightning import Callback
from collections import Iterable

import torch
import omegaconf.omegaconf
import typing
import logging
import toolz
import functools
import numpy as np

log = logging.getLogger(__name__)

__all__ = ['Classification']

def _safe_get_list(
    list: typing.Sequence[typing.Any], 
    index: int=0,
) -> typing.Any:
    return next(iter(list[index:index+1]), None)

def _binary(
    classified:     torch.Tensor,
    index:          int,
):
    return classified[index]

def _total(
    classified:     torch.Tensor,
):
    return torch.sum(classified)

class Classification(Metrics, Callback):
    execs: typing.List[typing.Callable] = []

    _MODES_ = {
        'binary': _binary,
        'total': _total,
    }

    def __init__(self, 
        gt:         typing.Sequence[str]=['label'],
        pred:       typing.Sequence[str]=['prediction'],
        weights:    typing.Sequence[str]=[],
        mask:       typing.Sequence[str]=[],
        # classes:    typing.Union[str, typing.Sequence[str]]='',
        mode:       typing.Union[int, str]='total',
        metrics:    omegaconf.DictConfig={},
    ):
        super(Classification, self).__init__(
            metrics, 
            **dict((k, {
                'gt': gt,
                'pred': pred,
                'weights': weights,
                'mask': mask
            }) for k in metrics.keys())
        )
        self.valid = True
        self.gt, self.pred, self.weights, self.mask = gt, pred, weights or [], mask or []
        # if isinstance(classes, str): #TODO: add loading cases (e.g. np/json files)
        #     with open(classes) as f:
        #         self.classes = [line.rstrip() for line in f]
        # elif isinstance(classes, Iterable):
        #     self.classes = classes
        # elif isinstance(classes, int):
        #     self.classes = ['' for _ in range(classes)]
        # else:
        #     log.error(f"The 'classes' parameter can only be one of [filepath, string list, int], instead it is {type(classes)}.")
        self.mode = Classification._MODES_[mode] if isinstance(mode, str) and mode == 'total'\
            else functools.partial(Classification._MODES_['binary'], index=mode)
        self.measures = { 'TP':0, 'TN': 0, 'FP': 0, 'FN': 0 }
        self.TP = TruePositive()
        self.TN = TrueNegative()
        self.FP = FalsePositive()
        self.FN = FalseNegative()
        self.metrics = None

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        metrics = { }
        for exe in self.execs:
            exe(tensors, metrics)
        returned = { }
        for k, m in metrics.items():
            returned[f'{k}'] = m
        for i, (gt, pred) in enumerate(zip(self.gt, self.pred)):
            w = _safe_get_list(self.weights, i)
            m = _safe_get_list(self.mask, i)
            for f in self.measures.keys():
                returned[f] = self.mode(
                    getattr(self, f)(
                        gt=tensors[gt],
                        pred=tensors[pred],
                        weights=tensors[w] if w else None,
                        mask=tensors[m] if m else None
                    )
                )
        if self.metrics:
            returned = toolz.merge(returned, self.metrics)
            self.metrics = None
        return returned

    def on_sanity_check_start(self, 
        trainer:    pytorch_lightning.Trainer,
        model:      pytorch_lightning.LightningModule,
    ) -> None:
        self.valid = False

    def on_sanity_check_end(self, 
        trainer:    pytorch_lightning.Trainer,
        model:      pytorch_lightning.LightningModule,
    ) -> None:
        self.valid = True

    # def on_validation_epoch_start(self, trainer, model):
    #     pass

    # def on_validation_batch_start(self, 
    #     trainer, model, batch, batch_idx, dataloader_idx
    # ):
    #     pass

    def on_validation_batch_end(self, 
        trainer:        pytorch_lightning.Trainer,
        model:          pytorch_lightning.LightningModule,
        outputs:        dict,
        batch:          dict,
        batch_idx:      int,
        dataloader_idx: int,
    ) -> None:
        if self.valid:
            self.measures['TP'] += int(outputs['TP'])
            self.measures['TN'] += int(outputs['TN'])
            self.measures['FP'] += int(outputs['FP'])
            self.measures['FN'] += int(outputs['FN'])

    # def on_validation_epoch_end(self, trainer, model):
    #     pass

    def on_validation_start(self, 
        trainer:    pytorch_lightning.Trainer,
        model:      pytorch_lightning.LightningModule,
    ) -> None:
        self.measures = { 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0 }

    def on_validation_end(self, 
    # def on_validation_epoch_end(self,     
        trainer:    pytorch_lightning.Trainer,
        model:      pytorch_lightning.LightningModule,
    ) -> None:
        if not self.valid:
            return
        tp = np.float32(self.measures['TP'])
        tn = np.float32(self.measures['TN'])
        fp = np.float32(self.measures['FP'])
        fn = np.float32(self.measures['FN'])
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        neg_pred_value = tn / (tn + fn)
        threat_score = tp / (tp + fn + fp)
        f1 = (2 * prec * recall) / (prec + recall)
        metrics = {
            'precision': prec,
            'recall': recall,
            'specificity': specificity,
            'negative_predictive_value': neg_pred_value,
            'threat_score': threat_score,
            'F1': f1,
        }
        model.log_dict(metrics, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
        log_metrics = toolz.keymap(lambda k: f"val_{k}", metrics)
        model.log_dict(log_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        self.metrics = metrics

        