import torch
import numpy as np
import os
import logging
import pickle

__all__ = ['JointRegressor']

log = logging.getLogger(__name__)

class JointRegressor(torch.nn.Module):
    def __init__(self,
        weights_path:       str,
        persistence:        bool=True,
    ) -> None:
        super().__init__()
        name, ext = os.path.splitext(os.path.basename(weights_path))
        if ext == '.npz':
            weights = np.load(weights_path)
        elif ext == '.pkl':
            with open(weights_path, 'rb') as f:
                weights = pickle.load(f, encoding='latin1')
        else:
            log.error(f"Unsupported extension ({ext}) for loading the joint regressor weights @ {weights_path}.")
        self.register_buffer('weights', 
            torch.from_numpy(weights).float(),
            persistent=persistence
        )

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bvc,jv->bjc', vertices, self.weights)