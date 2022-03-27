from moai.utils.color.colorize import get_colormap, COLORMAPS
from moai.visualization.visdom.base import Base

import torch
import visdom
import numpy as np
import functools
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Blend2d"]

class Blend2d(Base):
    def __init__(self,
        left:           typing.Union[str, typing.Sequence[str]],
        right:          typing.Union[str, typing.Sequence[str]],
        blending:       typing.Union[float, typing.Sequence[float]],
        colormap:       typing.Union[str, typing.Sequence[str]],
        transform:      typing.Union[str, typing.Sequence[str]],
        scale:          float=1.0,
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097,   
    ):
        super(Blend2d, self).__init__(name, ip, port)                
        self.left = [left] if type(left) is str else list(left)
        self.right = [right] if type(right) is str else list(right)
        self.blending = [blending] if type(blending) is float else list(blending)
        self.transforms = [transform] if type(transform) is str else list(transform)
        self.colormaps = [colormap] if type(colormap) is str else list(colormap)
        self.scale = scale
        self.transform_map = {
            'none': functools.partial(self.__no_transform),
            'minmax': functools.partial(self.__minmax_normalization),
            'sum_minmax': functools.partial(self.__sum_minmax_norm)
        }
        self.colorize_map = { "none": lambda x: x }
        self.colorize_map.update(COLORMAPS)

    @property
    def name(self) -> str:
        return self.env_name
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for l, r, b, t, c in zip(self.left, self.right, self.blending, 
            self.transforms, self.colormaps):
                n = l + "_" + r
                left = tensors[l]
                left = left.cpu().detach().numpy() if left.is_cuda else left.detach().numpy()                
                self.__viz_color(self.visualizer,
                    left * b + (1.0 - b) * self.colorize_map[c](
                        self.transform_map[t](tensors[r])), n, n, self.name,
                    self.scale
                )

    @staticmethod
    def __viz_color(
        visdom: visdom.Visdom,
        array:  np.ndarray,
        key:    str,
        win:    str,
        env:    str,
        scale:  float,
    ) -> None:
        if scale != 1.0:
            array = torch.nn.functional.interpolate(
                torch.from_numpy(array), mode='bilinear',
                scale_factor=scale, 
            ).numpy()
        visdom.images(
            np.clip(array, 0.0, 1.0),
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        )

    @staticmethod #TODO: refactor these into a common module
    def __no_transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod #TODO: refactor these into a common module
    def __minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
        b, _, __, ___ = tensor.size()
        min_v = torch.min(tensor.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        max_v = torch.max(tensor.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        return (tensor - min_v) / (max_v - min_v)

    @staticmethod #TODO: refactor these into a common module
    def __sum_minmax_norm(tensor: torch.Tensor) -> torch.Tensor:
        b, _, __, ___ = tensor.size()
        aggregated = torch.sum(tensor, dim=1, keepdim=True)
        min_v = torch.min(aggregated.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        max_v = torch.max(aggregated.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        return (aggregated - min_v) / (max_v - min_v)
            


