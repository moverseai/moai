import torch


class Gather2d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Gather2d, self).__init__()

    def forward(
        self,
        kpts: torch.Tensor,  # ASSUME N 2D keypoints with N X 2 shape (H x W)
        im: torch.Tensor,  # ASSUME a (B x C) x H x W image
    ) -> torch.Tensor:
        if kpts.shape[-1] > 2:
            kpts = kpts[..., :2]
        if len(im.shape) <= 3:
            im = im.unsqueeze(0).clone()
        b, c, h, w = im.shape
        im_flat = im.flatten(start_dim=-2, end_dim=-1)
        kpts_flat = kpts.flatten(start_dim=0, end_dim=-2)
        kpts_idxs = w * kpts_flat[:, 1] + kpts_flat[:, 0]

        return torch.gather(
            im_flat, -1, kpts_idxs.unsqueeze(0).unsqueeze(0).long()
        )  # B X Features X Kpts
