import torch

from moai.utils.torch import cross_product


# TODO: produces an error when using odd batch size, check it out
class NormalEstimation2d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(NormalEstimation2d, self).__init__()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        v_pad = torch.nn.functional.pad(points, (0, 0, 0, 1), mode="replicate")
        u_pad = torch.cat([points, points[:, :, :, :1]], dim=-1)
        du = u_pad[:, :, :, :-1] - u_pad[:, :, :, 1:]
        dv = v_pad[:, :, :-1, :] - v_pad[:, :, 1:, :]
        normals = cross_product(dv, du, dim=1)
        return torch.nn.functional.normalize(normals)
