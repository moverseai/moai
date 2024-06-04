import functools
import logging
import os
import typing
from collections.abc import Callable

import numpy as np
import torch
import torchvision

import moai.utils.color.colorize as mic
from moai.utils.arguments import ensure_choices, ensure_path

log = logging.getLogger(__name__)

__all__ = ["Blend2d"]


class Blend2d(Callable):

    __MODES__ = ["overwrite", "all"]
    __FORMATS__ = ["jpg", "png", "exr"]

    def __init__(
        self,
        path: str,
        left: typing.Union[str, typing.Sequence[str]],
        right: typing.Union[str, typing.Sequence[str]],
        blending: typing.Union[float, typing.Sequence[float]],
        colormap: typing.Union[str, typing.Sequence[str]],
        transform: typing.Union[str, typing.Sequence[str]],
        mode: str = "overwrite",  # all"
        extensions: typing.Union[str, typing.Sequence[str]] = [
            "png"
        ],  # jpg or png or exr
        scale: float = 1.0,
    ):
        self.mode = ensure_choices(log, "saving mode", mode, Blend2d.__MODES__)
        folder = ensure_path(log, "output folder", path)
        self.folder = os.path.join(os.getcwd(), folder)
        self.formats = [
            ensure_choices(log, "output format", ext, Blend2d.__FORMATS__)
            for ext in extensions
        ]
        self.mode = mode
        self.scale = scale
        self.index = 0
        self.left = [left] if type(left) is str else list(left)
        self.right = [right] if type(right) is str else list(right)
        self.blending = [blending] if type(blending) is float else list(blending)
        self.transforms = [transform] if type(transform) is str else list(transform)
        self.colormaps = [colormap] if type(colormap) is str else list(colormap)
        self.transform_map = {
            "none": functools.partial(self.__no_transform),
            "minmax": functools.partial(self.__minmax_normalization),
            "sum_minmax": functools.partial(self.__sum_minmax_norm),
        }
        self.colorize_map = {"none": lambda x: x.cpu().numpy()}
        self.colorize_map.update(mic.COLORMAPS)
        log.info(
            f"Exporting blended images @ {self.folder}/[left]_[right]_index.[format]."
        )

    def __call__(
        self,
        tensors: typing.Dict[str, torch.Tensor],
        step: typing.Optional[int] = None,
    ) -> None:
        for l, r, b, t, c, f in zip(
            self.left,
            self.right,
            self.blending,
            self.transforms,
            self.colormaps,
            self.formats,
        ):
            n = l + "_" + r
            left = tensors[l]
            left = (
                left.cpu().detach().numpy() if left.is_cuda else left.detach().numpy()
            )
            bs = self._save_color(
                self.folder,
                left * b
                + (1.0 - b) * self.colorize_map[c](self.transform_map[t](tensors[r])),
                n,
                self.index,
                f,
                self.scale,
            )
        self.index = 0 if self.mode == "overwrite" else self.index + bs

    @staticmethod
    def _save_color(
        path: str,
        array: np.ndarray,
        key: str,
        index: int,
        ext: str,
        scale: float,
    ) -> None:
        b, _, __, ___ = array.shape
        for i in range(b):
            img = torch.from_numpy(array[i, ...])
            if scale != 1.0:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    scale_factor=scale,
                    mode="bilinear",
                ).squeeze(0)
            torchvision.utils.save_image(img, f"{key}_{index + i}.{ext}")
        return b

    @staticmethod  # TODO: refactor these into a common module
    def __no_transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod  # TODO: refactor these into a common module
    def __minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
        b, _, __, ___ = tensor.size()
        min_v = (
            torch.min(tensor.view(b, -1), dim=1, keepdim=True)[0]
            .unsqueeze(2)
            .unsqueeze(3)
        )
        max_v = (
            torch.max(tensor.view(b, -1), dim=1, keepdim=True)[0]
            .unsqueeze(2)
            .unsqueeze(3)
        )
        return (tensor - min_v) / (max_v - min_v)

    @staticmethod  # TODO: refactor these into a common module
    def __sum_minmax_norm(tensor: torch.Tensor) -> torch.Tensor:
        b, _, __, ___ = tensor.size()
        aggregated = torch.sum(tensor, dim=1, keepdim=True)
        min_v = (
            torch.min(aggregated.view(b, -1), dim=1, keepdim=True)[0]
            .unsqueeze(2)
            .unsqueeze(3)
        )
        max_v = (
            torch.max(aggregated.view(b, -1), dim=1, keepdim=True)[0]
            .unsqueeze(2)
            .unsqueeze(3)
        )
        return (aggregated - min_v) / (max_v - min_v)
