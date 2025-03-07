import logging
import typing

import torch

from moai.utils.arguments.choices import ensure_choices

__all__ = ["Remap"]


log = logging.getLogger(__name__)


class Remap(torch.nn.Module):
    __PAD_MODES__ = ["zeros", "border", "reflection"]
    __SAMPLE_MODES__ = ["bilinear", "nearest", "bicubic"]

    def __init__(
        self,
        sample_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = None,
    ):
        super().__init__()
        self.sample_mode = ensure_choices(
            log, "sample mode", sample_mode, Remap.__SAMPLE_MODES__
        )
        self.padding_mode = ensure_choices(
            log, "padding mode", padding_mode, Remap.__PAD_MODES__
        )
        self.align_corners = align_corners

    def forward(self, tensor: torch.Tensor, map: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            tensor, map, self.sample_mode, self.padding_mode, self.align_corners
        )
