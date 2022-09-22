from moai.visualization.visdom.base import Base
from moai.monads.execution.cascade import _create_accessor

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import torch
import seaborn
import typing
import logging
import numpy as np
import pytorch_lightning

log = logging.getLogger(__name__)

__all__ = ["Density2d"]

class Density2d(Base, pytorch_lightning.Callback):
    def __init__(self,
        key:                typing.Union[str, typing.Sequence[str]],
        fill:               bool=True,
        palette:            str='coral',
        levels:             int=10,
        width:              int=720,
        height:             int=480,
        name:               str="default",
        ip:                 str="http://localhost",
        port:               int=8097,
    ):
        super().__init__(name, ip, port)
        self.levels = levels
        self.fill = fill
        self.palette = palette
        self.width = width / 300.0
        self.height = height / 300.0
        self.names = [key] if isinstance(key, str) else list(key)
        self.keys = [_create_accessor(k) for k in self.names]
        self.cache = {}

    @property
    def name(self) -> str:
        return self.env_name
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None
    ) -> None:
        for n, k in zip(self.names, self.keys):
            self.cache[n] = k(tensors)
            
    def on_train_epoch_end(self, 
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        for n in self.names:
            fig = Figure(figsize=(self.width, self.height), dpi=300.0)#self.dpi)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            seaborn.kdeplot(
                x=self.cache[n][:, 0], 
                y=self.cache[n][:, 1],
                fill=self.fill, palette=self.palette,
                ax=ax, levels=self.levels,
            )
            canvas.draw()
            img = np.asarray(canvas.buffer_rgba())
            self.visualizer.images(
                img[np.newaxis, ..., :3].transpose(0, 3, 1, 2),
                win=n, env=self.name
            )
            