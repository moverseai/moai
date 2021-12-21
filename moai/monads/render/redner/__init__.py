import pytorch_lightning
from pytorch_lightning.callbacks import Callback
import pyredner
import typing
import torch

class Redner(Callback): #TODO: need to update all engines to be the same hydra node, this will allow for a single engine config entry for this one
    def __init__(self,
        print_timings:          bool=False,
    ):
        self.timings = print_timings

    def on_fit_start(self, 
        trainer: pytorch_lightning.Trainer, 
        pl_module: pytorch_lightning.LightningModule,
        stage: typing.Optional[str]=None
    ) -> None:
        pyredner.set_print_timing(self.timings)
        pyredner.set_device(pl_module.device)
        pyredner.set_use_gpu(pl_module.device != torch.device('cpu'))

from moai.monads.render.redner.mesh import Silhouette as MeshSilhouette

__all__ = [
    'Redner',
    'MeshSilhouette',
]