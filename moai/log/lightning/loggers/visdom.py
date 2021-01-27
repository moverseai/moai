from moai.utils.color.colormap import colormap

import numpy
import visdom
import pytorch_lightning
import logging
import typing
import toolz
from json2html import *

log = logging.getLogger(__name__)

__all__ = ["Visdom"]

class Visdom(pytorch_lightning.loggers.base.LightningLoggerBase):
    def __init__(self,
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097,
        version:        int=0,
        train_key:      str="train_loss",
        test_key:       str="test_results",
        val_key:        str="val_metrics",
        plots_key:      str="plot",
        epoch_key:      str="epoch",
        step_key:       str="global_step",
        clear_window:   bool=True
    ):
        super(Visdom, self).__init__()
        # lighting logger
        self._rank, self._version = 0, version
        # visdom
        self.env_name, self.ip, self.port, self.viz = name, ip, port, None 
        # keys  
        self.keys: typing.Dict[str, str]={ }
        self.keys.update({
            'train': train_key, 'val': val_key, 'test': test_key,
            'loss_plots': plots_key, 'epoch': epoch_key, 'step': step_key
        })
        # plotting
        self.colors = colormap().astype(numpy.int32)
        # other
        self.colors: numpy.array
        self.best_epoch_loss: typing.List[float]=[]
        self.best_epoch_step: typing.List[int]=[]
        self.best_epoch_val_loss: typing.Dict[str, typing.List[float]]={ }
        self.best_epoch_val_step: typing.Dict[str, typing.List[int]]={ }
        self.loss_plots: typing.Dict[str, str]={ }
        self.metric_plots: typing.Dict[str, str]={ }
        self.clear_window = clear_window

    @property
    def name(self) -> str:
        return self.env_name

    @property
    def visualizer(self) -> visdom.Visdom:
        if self.viz is not None:
            return self.viz

        self.viz = visdom.Visdom(
            server=self.ip,
            port=self.port,
            env=self.name,
            use_incoming_socket=False
        )
        return self.viz

    def _plot_train_loss(self,
        epoch: int,
        value: float,
        step: int
    ) -> None:
        if not self.keys['train'] in self.loss_plots.keys():
            self.loss_plots[self.keys['train']] = self.visualizer.line(
                X=numpy.array([step,step]),
                Y=numpy.array([value,value]),
                env=self.name,
                name=self.keys['train'],
                opts={
                    'linecolor': numpy.array([self.colors[0]]), # first color
                    # 'legend': [str(self.keys['train']), str(self.keys['epoch'])],
                    'legend': [str(self.keys['train']), 'train_' + str(self.keys['epoch'])],
                    'title': str(self.keys['train']),
                    'xlabel':'steps',
                    'ylabel': str(self.keys['train'])
                }
            )
        else:
            self.visualizer.line(
                X=numpy.array([step]),
                Y=numpy.array([value]),
                win=self.loss_plots[self.keys['train']],
                name=self.keys['train'], 
                update='append',
            )
        while len(self.best_epoch_loss) <= epoch:
            self.best_epoch_loss.append(numpy.Infinity)
            self.best_epoch_step.append(step)        
        if value < self.best_epoch_loss[epoch]:
            self.best_epoch_loss[epoch] = value
            self.best_epoch_step[epoch] = step
            self.visualizer.line(
                X=numpy.array(self.best_epoch_step),
                Y=numpy.array(self.best_epoch_loss),
                update='insert',
                win=self.loss_plots[self.keys['train']],
                name='train_' + self.keys['epoch'],
                opts={
                    'markers': True,
                    'linecolor': numpy.array([self.colors[-1]]) # last color
                }
            )

    def _plot_losses(self,
        losses:         typing.Dict[str, float],
        epoch:          int,
        step_num:       int,
    ) -> None:
        for i, (k, v) in enumerate(losses.items()):
            if not self.keys['loss_plots'] in self.loss_plots.keys():            
                self.loss_plots[self.keys['loss_plots']] = self.visualizer.line(
                    X=numpy.array([step_num,step_num]),
                    Y=numpy.array([v,v]),
                    env=self.name,
                    name=k,
                    opts={
                        'linecolor': numpy.array([self.colors[2 + i]]), # second color and on
                        'legend': [k for k in losses.keys()],
                        'title': self.keys['loss_plots'],
                        'xlabel': 'steps',
                        'ylabel': 'loss plots'
                    }
                )
            else:
                self.visualizer.line(
                    X=numpy.array([step_num]),
                    Y=numpy.array([v]),
                    win=self.loss_plots[self.keys['loss_plots']],
                    name=k, 
                    update='append',
                )

    def _plot_val_loss(self,
        metrics: typing.Dict[str, typing.Any],
        epoch: int,
        step_num: int
    ) -> None:
        for i, key in enumerate(metrics.keys()):
            value = metrics[key]
            if key not in self.metric_plots.keys():
                self.metric_plots[key] = self.visualizer.line(
                    X=numpy.array([step_num,step_num]),
                    Y=numpy.array([value,value]),
                    env=self.name,
                    name=key,
                    opts={
                        'linecolor': numpy.array([self.colors[2 + i]]), # second color and on
                        'legend': [k for k in metrics.keys() if k == key],
                        'title': key,
                        'xlabel':'steps',
                        'ylabel': key
                    }
                )
            else:
                self.visualizer.line(
                    X=numpy.array([step_num]),
                    Y=numpy.array([value]),
                    win=self.metric_plots[key],
                    name=key, 
                    update='append',
                )
            if key not in self.best_epoch_val_loss.keys():
                self.best_epoch_val_loss[key] = []
                self.best_epoch_val_step[key] = []
            while len(self.best_epoch_val_loss[key]) <= epoch:
                self.best_epoch_val_loss[key].append(numpy.Infinity)
                self.best_epoch_val_step[key].append(step_num)
            if value < self.best_epoch_val_loss[key][epoch]:
                self.best_epoch_val_loss[key][epoch] = value
                self.best_epoch_val_step[key][epoch] = step_num
                ind = numpy.isfinite(self.best_epoch_val_loss[key])
                self.visualizer.line(
                    X=numpy.array(self.best_epoch_val_step[key])[ind],
                    Y=numpy.array(self.best_epoch_val_loss[key])[ind],
                    update='insert',
                    win=self.metric_plots[key],
                    name='val_' + self.keys['epoch'],
                    opts={
                        'markers': True,
                        'linecolor': numpy.array([self.colors[2 + i]])
                    }
                )

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
        if train_metrics:
            epoch = int(metrics['epoch'])
            loss = float(metrics['total_loss'])
            self._plot_train_loss(epoch, loss, step)
            self._plot_losses(train_metrics, epoch, step)
        elif 'test' in metrics:
            return #TODO: test case 
        if val_metrics:
            epoch = int(metrics['epoch'])
            self._plot_val_loss(val_metrics, epoch, step)

    @pytorch_lightning.loggers.base.rank_zero_only
    def log_hyperparams(self,
        params: typing.Dict[str, typing.Any] #TODO: or namespace object ?
    ) -> None:
        """Record hyperparameters
            :param params: argparse.Namespace containing the hyperparameters
        """
        self.visualizer.text(json2html.convert(json=dict(params)))

    @pytorch_lightning.loggers.base.rank_zero_only
    def save(self) -> None:
        """Save log data"""
        # self.visualizer.save(self.name)
        pass #TODO: support saving ?

    @pytorch_lightning.loggers.base.rank_zero_only
    def finalize(self, 
        status: str
    ) -> None:
        """Do any processing that is necessary to finalize an experiment
            :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()
        if self.clear_window:
            self.visualizer.close()

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