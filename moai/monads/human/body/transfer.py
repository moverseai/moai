import pickle
import torch
import os
import numpy as np
import logging

log = logging.getLogger(__name__)

__all__ = ['BodyTransfer']

class BodyTransfer(torch.nn.Module):
    def __init__(self,
        weights_path:   str,
        persistence:    bool=True,
    ) -> None:
        super().__init__()
        name, ext = os.path.split(weights_path)
        if ext == '.pkl':
            with open(weights_path, 'rb') as f:
                w = pickle.load(f, encoding='latin1')['mtx']
                if hasattr(w, 'todense'):
                    w = w.todense()
                w = w[:, :w.shape[-1] // 2]
        elif ext == '.npz' or ext == '.npy':
            w = np.load(weights_path)
        else:
            log.error(f"Unsupported extension ({ext}) to load body tranfer weights from when loading {weights_path}. Can only load *.pkl, *.npy, *.npz files")
        self.register_buffer('weights', torch.from_numpy(w).float(), persistent=persistence)

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bni,mn->bmi', vertices, self.weights)