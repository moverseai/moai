import torch

class Deprojection(torch.nn.Module):
    def __init__(self,
    ):
        super(Deprojection, self).__init__()

    def forward(self,
        intrinsics:         torch.Tensor,
        depthmap:           torch.Tensor,
        grid:               torch.Tensor=None,
    ) -> torch.Tensor:
        b, _, h, w = depthmap.shape
        ones = torch.ones(depthmap.shape, device=depthmap.device)
        pseudo3d = torch.cat((grid, ones), dim=-3).view(b, 3, -1)
        p3d = torch.einsum(
                'bij,bjk->bik', torch.inverse(intrinsics), pseudo3d
            ).reshape(b, 3, h, w) * depthmap

        return p3d
