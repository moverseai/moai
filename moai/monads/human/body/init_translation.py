import torch
import typing

#NOTE: code from https://github.com/vchoutas/smplify-x

__all__ = ["InitTranslation"]

class InitTranslation(torch.nn.Module):
    def __init__(self,
        torso_edge_indices: typing.Sequence[typing.Tuple[int, int]]=[(5, 12), (2, 9)],
        focal_length:       float=5000.0,
    ):
        super(InitTranslation, self).__init__()
        self.torso_edge_indices = torso_edge_indices
        self.focal_length = float(focal_length)
        
    def forward(self, 
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
    ) -> torch.Tensor:
        diff3d = []
        diff2d = []
        for edge in self.torso_edge_indices:
            diff3d.append(joints3d[:, edge[0]] - joints3d[:, edge[1]])
            diff2d.append(joints2d[:, edge[0]] - joints2d[:, edge[1]])

        diff3d = torch.stack(diff3d, dim=1)
        diff2d = torch.stack(diff2d, dim=1)

        length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
        length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

        height2d = length_2d.mean(dim=1)
        height3d = length_3d.mean(dim=1)

        est_d = self.focal_length * (height3d / height2d)

        b = joints3d.shape[0]        
        return torch.cat([
            torch.zeros([b, 2], device=est_d.device), est_d.expand(b, 1)
        ], dim=1)