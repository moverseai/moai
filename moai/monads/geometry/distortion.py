import torch
import numpy as np

__all__ = ['Distort']

class Distort(torch.nn.Module):
    def __init__(self,
        epsilon:        float=1e-8,
    ) -> None:
        super().__init__()
        self.eps = epsilon

    def forward(self,
        points:             torch.Tensor,
        coefficients:       torch.Tensor,
    ) -> torch.Tensor:
        coefficients_remainder = 8 - coefficients.shape[-1]
        if coefficients_remainder:
            coefficients = torch.cat([
                coefficients, torch.zeros_like(coefficients[..., :coefficients_remainder])
            ], dim=-1)
        k1 = coefficients[..., 0][:, np.newaxis, np.newaxis]
        k2 = coefficients[..., 1][:, np.newaxis, np.newaxis]
        p1 = coefficients[..., 2][:, np.newaxis, np.newaxis]
        p2 = coefficients[..., 3][:, np.newaxis, np.newaxis]
        k3 = coefficients[..., 4][:, np.newaxis, np.newaxis]
        k4 = coefficients[..., 5][:, np.newaxis, np.newaxis]
        k5 = coefficients[..., 6][:, np.newaxis, np.newaxis]
        k6 = coefficients[..., 7][:, np.newaxis, np.newaxis]        
        x, y = torch.split(points[..., :2] / (points[..., 2:] + self.eps), 1, dim=-1)
        x_sq = torch.pow(x, 2)
        y_sq = torch.pow(y, 2)
        xy = x * y
        r_sq = x_sq + x_sq
        c1 = 1 + r_sq * (k1 + r_sq * (k2 + r_sq * k3))
        c2 = 1 + r_sq * (k4 + r_sq * (k5 + r_sq * k6))
        c =	c1 / c2
        x_d = c * x + p1 * 2.0 * xy + p2 * (r_sq + 2 * x_sq)
        y_d = c * y + p2 * 2.0 * xy + p1 * (r_sq + 2 * y_sq)
        z_d = points[..., 2:3]
        return torch.cat([x_d * z_d, y_d * z_d, z_d], dim=-1)
