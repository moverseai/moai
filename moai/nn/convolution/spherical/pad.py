import functools

import torch

__all__ = [
    "SphericalPad2d",
]

# NOTE: from https://github.com/pytorch/pytorch/issues/3858
# NOTE: check https://github.com/pytorch/pytorch/issues/20981
# also https://github.com/pytorch/pytorch/issues/3858#issuecomment-432801360


def __pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """
    if isinstance(dim, int):
        dim = [dim]
    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")
        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)
        idx = tuple(
            slice(None if s != d else -2 * pad, None if s != d else -pad, 1)
            for s in range(len(x.shape))
        )
        x = torch.cat([x[idx], x], dim=d)
        pass
    return x


horizontal_circular_pad2d = functools.partial(__pad_circular_nd, dim=[3])


class SphericalPad2d(torch.nn.Module):
    def __init__(
        self,
        padding: int = 1,
    ):
        super(SphericalPad2d, self).__init__()
        self.padding = padding

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # assumes [B, C, H, W] tensor inputs
        return torch.nn.functional.pad(
            horizontal_circular_pad2d(x, pad=self.padding),
            pad=[0, 0, self.padding, self.padding],
            mode="replicate",
        )
