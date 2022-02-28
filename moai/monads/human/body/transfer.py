import pickle
import torch

__all__ = ['BodyTransfer']

class BodyTransfer(torch.nn.Module):
    def __init__(self,
        weights_path:   str,
        persistence:    bool=True,
    ) -> None:
        super().__init__()
        with open(weights_path, 'rb') as f:
            w = pickle.load(f, encoding='latin1')['mtx']
            if hasattr(w, 'todense'):
                w = w.todense()
            w = w[:, :w.shape[-1] // 2]
        self.register_buffer('weights', torch.from_numpy(w).float(), persistent=persistence)

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bni,mn->bmi', vertices, self.weights)