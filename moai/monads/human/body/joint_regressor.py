import torch
import numpy as np

__all__ = ['JointRegressor']

class JointRegressor(torch.nn.Module):
    def __init__(self,
        weights_path:       str,
        persistence:         bool=True,
    ) -> None:
        super().__init__()
        self.register_buffer('weights', 
            torch.from_numpy(np.load(weights_path)).float(),
            persistent=persistence
        )

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bvc,jv->bjc', vertices, self.weights)