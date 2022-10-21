from pytorch_lightning.callbacks import Callback
import numpy as np
import logging
import torch

log = logging.getLogger(__name__)

__all__ = ["LatentInterp"]

class LatentInterp(Callback):
    def __init__(self,
        interpolate_epoch_interval: int,
        generated: str,
        range_start: int=-5,
        range_end: int=5,
        steps: int=11,
        num_samples: int=1,
        full_vector: bool=False,
    ):
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.key = generated
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.steps = steps
        self.full_vector = full_vector

    def on_validation_epoch_start(self, trainer, pl_module):
        points = self.interpolate_latent(pl_module, pl_module.latent_dim)
        td = {'__moai__': {'epoch': trainer.current_epoch}, self.key: points}
        td = pl_module.generation(td)
        for vis in pl_module.visualization.latent_visualizers:
            vis(td)

    def interpolate_latent(self, pl_module, latent_dim):
        with torch.no_grad():
            pl_module.eval()
            points_list = []
            if self.full_vector:
                z = torch.randn(self.num_samples, latent_dim, device=pl_module.device)
                for full_z in np.linspace(self.range_start, self.range_end, self.steps):
                    z[:] += torch.tensor(full_z)
                    points = pl_module.decoder(z) 
                    points_list.append(points)
            else:
                z = torch.randn(self.num_samples, latent_dim, device=pl_module.device)
                for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                    for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                        # set the fist 2 dims to the value
                        z[:, 0] = torch.tensor(z1)
                        z[:, 1] = torch.tensor(z2)
                        # generate
                        points = pl_module.decoder(z)
                        points_list.append(points)
        return torch.stack(points_list, 1).squeeze(0)