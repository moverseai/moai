from moai.visualization.visdom.base import Base
from moai.utils.color.colorize import get_colormap, COLORMAPS
from moai.utils.arguments import assert_numeric

import torch
import visdom
import functools
import typing
import logging
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Image2d"]

class Image2d(Base):
    def __init__(self,
        keys:               typing.Union[str, typing.Sequence[str]],
        types:              typing.Union[str, typing.Sequence[str]],
        colormaps:          typing.Union[str, typing.Sequence[str]],
        transforms:         typing.Union[str, typing.Sequence[str]],
        batch_percentage:   float=1.0,
        name:               str="default",
        ip:                 str="http://localhost",
        port:               int=8097,   
    ):
        super(Image2d, self).__init__(name, ip, port)
        self.keys = [keys] if type(keys) is str else list(keys)
        self.types = [types] if type(types) is str else list(types)
        self.transforms = [transforms] if type(transforms) is str else list(transforms)
        self.colormaps = [colormaps] if type(colormaps) is str else list(colormaps)
        self.batch_percentage = batch_percentage
        assert_numeric(log, 'batch percentage', batch_percentage, 0.0, 1.0)
        self.viz_map = {
            'color': functools.partial(self.__viz_color, self.visualizer),
            'heatmap': functools.partial(self.__viz_heatmap, self.visualizer),
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
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for k, t, tf, c in zip(self.keys, self.types, self.transforms, self.colormaps):
            self.viz_map[t](
                self.colorize_map[c](
                    self.transform_map[tf](
                        tensors[k][:int(self.batch_percentage * tensors[k].shape[0])]
                    )
                ), k, k, self.name
            )

    @staticmethod
    def __viz_color(
        visdom: visdom.Visdom,
        tensor: np.array,
        key: str,
        win: str,
        env: str
    ) -> None:
        visdom.images(
            tensor,
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        )

    @staticmethod
    def __viz_heatmap(
        visdom: visdom.Visdom,
        tensor: torch.Tensor,
        key: str,
        win: str,
        env: str
    ) -> None:
        b, _, __, ___ = tensor.size() #NOTE: assumes [B, C, H, W], i.e. 2d train
        heatmaps = torch.flip(tensor, dims=[2]).detach().cpu()
        for i in range(b):
            opts = (
            {
                'title': key + "_" + str(i),
                'colormap': 'Viridis'
            })
            visdom.heatmap(heatmaps[i, :, :, :].squeeze(), opts=opts, win=win + str(i))

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
    def __ndc_to_one(tensor: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(
            torch.scalar_tensor(0.5).to(tensor), 
            tensor,
            torch.scalar_tensor(0.5).to(tensor)
        )