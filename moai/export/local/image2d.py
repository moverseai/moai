from collections.abc import Callable
from moai.utils.arguments import (
    ensure_path,
    ensure_choices,
)

import moai.utils.color.colorize as mic

import os
import functools
import numpy as np
import torch
import torchvision
import typing
import logging
import cv2

__all__ = ["Image2d"]

log = logging.getLogger(__name__)

class Image2d(Callable):

    __MODES__ = ['overwrite', 'all']
    __FORMATS__ = ['jpg', 'png', 'exr']

    def __init__(self,
        path:       str,
        image:      typing.Union[str, typing.Sequence[str]],
        type:       typing.Union[str, typing.Sequence[str]],
        colormap:   typing.Union[str, typing.Sequence[str]],
        transform:  typing.Union[str, typing.Sequence[str]],
        mode:       str="overwrite", # all"
        extension:  typing.Union[str, typing.Sequence[str]]=["png"], # jpg or png or exr
    ):
        self.mode = ensure_choices(log, "saving mode", mode, Image2d.__MODES__)
        folder = ensure_path(log, "output folder", path)
        self.formats = [ensure_choices(log, "output format", ext, Image2d.__FORMATS__) for ext in extension]
        self.mode = mode
        self.index = 0
        self.keys = [image] if isinstance(image, str) else list(image)
        self.types = [type] if isinstance(type, str) else list(type)
        self.transforms = [transform] if isinstance(transform, str) else list(transform)
        self.colormaps = [colormap] if isinstance(colormap, str) else list(colormap)
        self.save_map = {
            'color': functools.partial(self._save_color, os.path.join(os.getcwd(), folder)),
            'depth': functools.partial(self._save_depth, os.path.join(os.getcwd(), folder)),
        }
        self.transform_map = {
            'none': functools.partial(self._no_transform),
            'minmax': functools.partial(self._minmax_normalization),
            "positive_minmax": functools.partial(self._minmax_normalization, positive_only=True),
            "log_minmax": functools.partial(self._log_minmax_normalization),
        }
        self.colorize_map = { "none": lambda x: x.cpu().numpy() }
        self.colorize_map.update(mic.COLORMAPS)
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for k, t, tf, c, f in zip(
            self.keys,
            self.types,
            self.transforms,
            self.colormaps,
            self.formats,
        ):
            b = self.save_map[t](
                self.colorize_map[c](
                    self.transform_map[tf](tensors[k])
                ), k, self.index, f
            )
        self.index = 0 if self.mode == "overwrite" else self.index + b

    @staticmethod
    def _save_color(
        path:               str,
        array:              np.array,
        key:                str,
        index:              int,
        ext:                str,
    ) -> int:
        b, _, __, ___ = array.shape
        for i in range(b):
        #    torchvision.utils.save_image(
        #        torch.from_numpy(array)[i, :, :, :], 
        #        f'{key}_{index + i}.{ext}'
        #     )
            cv2.imwrite(
                f'{key}_{index + i}.{ext}',
                array[i, :, :, :].transpose(1, 2, 0),
            )
        return b

    @staticmethod
    def _save_depth(
        path:               str,
        array:              np.array,
        key:                str,
        index:              int,
        ext:                str,
    ) -> int:
        b, _, __, ___ = array.shape
        for i in range(b): #NOTE: change to cv2 to support .exr
            cv2.imwrite(
                f'{key}_{index + i}.{ext}',
                array[i, :, :, :].transpose(1, 2, 0),
            )
        #    torchvision.utils.save_image(
        #        torch.from_numpy(array)[i, :, :, :], 
        #        f'{key}_{index + i}.{ext}'
        #     )
        return b

    @staticmethod #TODO: refactor these into a common module
    def _no_transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod #TODO: refactor these into a common module
    def _minmax_normalization(tensor: torch.Tensor, positive_only=False) -> torch.Tensor:
        b, _, __, ___ = tensor.size()
        t = tensor
        if positive_only:
            t[t < 0.0] = 0.0
        min_v = torch.min(t.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        max_v = torch.max(t.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        return (t - min_v) / (max_v - min_v)        
    
    @staticmethod #TODO: refactor these into a common module
    def _log_minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
        b, _, __, ___ = tensor.size()
        log_tensor = torch.log(tensor + 1e-8)
        min_v = torch.min(log_tensor.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        max_v = torch.max(log_tensor.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        return (log_tensor - min_v) / (max_v - min_v)