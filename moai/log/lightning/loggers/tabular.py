from collections import defaultdict

import tablib
import toolz
import pytorch_lightning
import typing
import logging 

log = logging.getLogger(__name__)

__all__ = ["Tabular"]

class Tabular(pytorch_lightning.loggers.base.LightningLoggerBase):
    def __init__(self,
        name:           str="default",
        version:        int=0,
    ):
        super(Tabular, self).__init__()   
        self._version = version
        self.file_name = name
        self.train_logs = tablib.Dataset()
        self.val_logs = defaultdict(tablib.Dataset)
        self.test_logs = tablib.Dataset()
        self.train_headers_written = False
        self.val_headers_written = defaultdict(bool)
        self.test_headers_written = False
        self.val_headers = { }

    @property
    def name(self) -> str:
        return self.file_name

    def _append_train_losses(self,
        metrics:            typing.Dict[str, typing.Any],
        epoch:              int,
        step:               int,
    ) -> None:
        loss = metrics['total_loss']
        train_metrics = toolz.dissoc(metrics, 'train', 'epoch', 'total_loss')
        if self.train_logs.headers is None and not self.train_headers_written:
            self.train_logs.headers = list(toolz.concat([
                [str('epoch'), str('iteration'), str('total_loss')],
                [k for k in train_metrics.keys()]
            ]))
        self.train_logs.append(list(
            toolz.concat([
                [epoch, step, loss],
                train_metrics.values()
            ])
        ))
        
    def _append_val_loss(self, 
        dataset:        str,
        metrics:        typing.Dict[str, typing.Any],
        epoch:          int,
        step:           int,
    ) -> None:
        if self.val_logs[dataset].headers is None and not self.val_headers_written[dataset]:
            self.val_logs[dataset].headers = list(toolz.concat([
                [str('epoch'), str('iteration')],
                [k for k in metrics.keys()]
            ]))
            self.val_headers[dataset] = self.val_logs[dataset].headers
        self.val_logs[dataset].append(list(
            toolz.concat([
                [epoch, step],
                [metrics[k] for k in self.val_headers[dataset] if k in metrics]
            ])
        ))

    def _append_test_metrics(self, 
        metrics:        typing.Dict[str, typing.Any],
        step:           int,
    ) -> None:
        if self.test_logs.headers is None and not self.test_headers_written:
            self.test_logs.headers = list(toolz.concat([
                [str('iteration')],
                [k for k in metrics.keys()]
            ]))
            self.test_headers = self.test_logs.headers
        self.test_logs.append(list(
            toolz.concat([
                [step],
                [metrics[k] for k in self.test_headers if k in metrics]
            ])
        ))

    @pytorch_lightning.loggers.base.rank_zero_only
    def log_metrics(self, 
        metrics:        typing.Dict[str, typing.Any],
        step:           int
    ) -> None:
        train_metrics = toolz.keymap(lambda k: k.replace('train_', ''), 
            toolz.keyfilter(lambda k: k.startswith('train_'), metrics)
        )
        val_metrics = toolz.keymap(lambda k: k.replace('val_', ''), 
            toolz.keyfilter(lambda k: k.startswith('val_'), metrics)
        )
        test_metrics = toolz.keymap(lambda k: k.replace('test_', '').replace('/epoch_0', ''), 
            toolz.keyfilter(lambda k: k.startswith('test_'), metrics)
        )
        if train_metrics:
            self._append_train_losses(
                toolz.assoc(train_metrics, 'total_loss', metrics['total_loss']), 
                metrics['epoch'],
                step
            )
        elif test_metrics:            
            self._append_test_metrics(test_metrics, step)
            return
        if val_metrics:
            dataset_val_metrics = toolz.valmap(
                lambda v: toolz.keymap(lambda k: k.split('/')[0], dict(v)), 
                # toolz.groupby(lambda k: k[0].split('/')[-1], val_metrics.items())
                toolz.groupby(
                    lambda k: toolz.get(1, k[0].split('/'), 'metrics'),
                    val_metrics.items()
                )
            )
            for k, v in dataset_val_metrics.items():
                self._append_val_loss(k, v, metrics['epoch'], step)

    @pytorch_lightning.loggers.base.rank_zero_only
    def log_hyperparams(self,
        params: typing.Dict[str, typing.Any] #TODO or namespace object ?
    ) -> None:
        """Record hyperparameters
            :param params: argparse.Namespace containing the hyperparameters
        """
        data = tablib.Dataset()
        data.headers = [k for k in dict(params).keys()]
        data.append([v for v in dict(params).values()])
        with open(self.name + "_hparams.yaml", 'w') as f:
            f.write(data.export('yaml'))

    @pytorch_lightning.loggers.base.rank_zero_only
    def save(self) -> None:
        if self.train_logs.height:
            """Save train log data"""
            with open(self.name + "_train.csv", 'a', newline='') as f:
                f.write(self.train_logs.export('csv'))
            if not self.train_headers_written and self.train_logs.headers is not None:
                self.train_headers_written = True
            self.train_logs.wipe()        
        if self.val_logs:
            """Save val log data"""
            for k, d in self.val_logs.items():
                with open(f"{self.name}_{k}_val.csv", 'a', newline='') as f:
                    f.write(d.export('csv'))
                if not self.val_headers_written[k] and d.headers is not None:
                    self.val_headers_written[k] = True
                d.wipe()
        if self.test_logs.height:
            """Save test log data"""
            with open(self.name + "_test.csv", 'a', newline='') as f:
                f.write(self.test_logs.export('csv'))
            if not self.test_headers_written and self.test_logs.headers is not None:
                self.test_headers_written = True
            self.test_logs.wipe()

    @pytorch_lightning.loggers.base.rank_zero_only
    def finalize(self, 
        status: str
    ) -> None:
        """Do any processing that is necessary to finalize an experiment
            :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()
        self.close()

    @property
    def rank(self) -> int:
        """
            Process rank. In general, metrics should only be logged by the process
                with rank 0
        """
        return self._rank

    @rank.setter
    def rank(self, value: int) -> None:
        """Set the process rank"""
        self._rank = value

    @property
    def version(self) -> int:
        """Return the experiment version"""
        return self._version

    @property
    def experiment(self) -> typing.Any:
        return self.name