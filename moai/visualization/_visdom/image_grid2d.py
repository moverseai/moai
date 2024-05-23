from moai.visualization.visdom.base import Base
from moai.utils.color.colorize import COLORMAPS
from torchvision.utils import make_grid

import torch
import visdom
import functools
import typing
import logging
import numpy

log = logging.getLogger(__name__)

__all__ = ["Image_grid2d"]

class Image_grid2d(Base):
    def __init__(self,
        keys:           typing.Union[str, typing.Sequence[str]],
        types:          typing.Union[str, typing.Sequence[str]],
        colormaps:      typing.Union[str, typing.Sequence[str]],
        transforms:     typing.Union[str, typing.Sequence[str]],
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097,
        jpeg_quality:       int=50,
    ):
        super(Image_grid2d, self).__init__(name, ip, port)
        self.keys = [keys] if type(keys) is str else list(keys)
        self.types = [types] if type(types) is str else list(types)
        self.transforms = [transforms] if type(transforms) is str else list(transforms)
        self.colormaps = [colormaps] if type(colormaps) is str else list(colormaps)
        self.viz_map = {
            'color_grid': functools.partial(self.__viz_color, self.visualizer, jpeg_quality=jpeg_quality)
        }
        self.transform_map = {
            'none': functools.partial(self.__no_transform),
            'minmax': functools.partial(self.__minmax_normalization),
            'ndc': functools.partial(self.__ndc_to_one),
        }
        self.colorize_map = { "none": lambda x: x }
        self.colorize_map.update(COLORMAPS)

    @property
    def name(self) -> str:
        return self.env_name

    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None
    ) -> None:
        _, _, ch, w, h = tensors.shape
        for t, tf, c in zip(self.types, self.transforms, self.colormaps):
            self.viz_map[t](
                self.colorize_map[c](
                    self.transform_map[tf](
                        make_grid(tensors.reshape(-1, ch, w, h), nrow=7)
                    )
                ), 'color_grid', 'color_grid', self.name
            )

    @staticmethod
    def __viz_color(
        visdom: visdom.Visdom,
        tensor: torch.Tensor,
        key: str,
        win: str,
        env: str,
        jpeg_quality:   int=50,
    ) -> None:
        visdom.images(
            tensor,
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': jpeg_quality,
            }
        )

    @staticmethod
    def __no_transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def __minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
        _, __, ___ = tensor.size()
        min_v = torch.min(tensor)
        max_v = torch.max(tensor)
        return (tensor - min_v) / (max_v - min_v)

    @staticmethod
    def __ndc_to_one(tensor: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(
            torch.scalar_tensor(0.5).to(tensor), 
            tensor,
            torch.scalar_tensor(0.5).to(tensor)
        )