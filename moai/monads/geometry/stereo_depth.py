import torch


class DepthFromStereo(torch.nn.Module):
    def __init__(
        self,
        baseline: float,  # physical distance between cameras in mm
    ):
        super(DepthFromStereo, self).__init__()
        self.base = baseline

    def forward(
        self,
        im_disp: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        if len(im_disp.shape) < 4:
            im_disp = im_disp.unsqueeze(0).clone()

        return self.base * intrinsics[0][0][0] / im_disp
