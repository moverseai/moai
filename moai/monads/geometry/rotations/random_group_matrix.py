# @ https://github.com/pimdh/lie-vae/blob/master/lie_vae/lie_tools.py

import numpy as np
import torch


class RandomGroupMatrices(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _quaternions_to_group_matrix(self, q):
        """Normalises q and maps to group matrix."""
        q = q / q.norm(p=2, dim=-1, keepdim=True)
        r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        return torch.stack(
            [
                r * r - i * i - j * j + k * k,
                2 * (r * i + j * k),
                2 * (r * j - i * k),
                2 * (r * i - j * k),
                -r * r + i * i - j * j + k * k,
                2 * (i * j + r * k),
                2 * (r * j + i * k),
                2 * (i * j - r * k),
                -r * r - i * i + j * j + k * k,
            ],
            -1,
        ).view(*q.shape[:-1], 3, 3)

    def _random_quaternions(self, n):
        u1, u2, u3 = torch.rand(3, n)
        return torch.stack(
            (
                torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2),
                torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2),
                torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
                torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
            ),
            1,
        )

    def forward(self, n):
        return self._quaternions_to_group_matrix(self._random_quaternions(n.shape[0]))
