from pytorch_lightning.callbacks import Callback
import omegaconf.omegaconf
import numpy as np
import logging
import torch

log = logging.getLogger(__name__)

__all__ = ["LatentInterp"]

class LatentInterp(Callback):
    def __init__(self,
        interpolate_epoch_interval: int,
        range_start: int=-5,
        range_end: int=5,
        steps: int=11,
        num_samples: int=2,
        normalize: bool=True,
    ):
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps

    def on_train_epoch_end(self, trainer, pl_module):
        points = self.interpolate_latent(pl_module, pl_module.latent_dim)
        # pl_module.visualization.latent_visualizers[0](points)

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     pl_module.visualizer.latent_visualizers[0](torch.stack(tensors, 1))
    
    def interpolate_latent(self, pl_module, latent_dim):
        with torch.no_grad():
            pl_module.eval()
            points_list = []
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    # set all dims to zero
                    z = torch.zeros(self.num_samples, latent_dim, device=pl_module.device)

                    # set the fist 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # generate
                    z_reshaped = pl_module.reparametrizer.linear_to_dec(z)
                    #TODO solve the reshape problem
                    points = pl_module.decoder(z_reshaped)
                    points_list.append(points)
        return torch.stack(points_list, 1).squeeze(0)