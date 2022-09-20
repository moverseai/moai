from moai.visualization.visdom.base import Base
from moai.utils.color.colorize import COLORMAPS

import torch
import visdom
import functools
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Feature2d"]

class Feature2d(Base):
    def __init__(self,
        image:           typing.Union[str, typing.Sequence[str]],
        type:          typing.Union[str, typing.Sequence[str]],
        colormap:      typing.Union[str, typing.Sequence[str]],
        transform:     typing.Union[str, typing.Sequence[str]],
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097,   
        jpeg_quality:       int=50,
    ):
        super(Feature2d, self).__init__(name, ip, port)
        self.images = [image] if isinstance(image, str) else list(image)
        self.types = [type] if isinstance(type, str) else list(type)
        self.transforms = [transform] if isinstance(transform, str) else list(transform)
        self.colormaps = [colormap] if isinstance(colormap, str) is str else list(colormap)
        self.viz_map = {
            'color': functools.partial(self.__viz_color, self.visualizer, jpeg_quality=jpeg_quality),
            'heatmap': functools.partial(self.__viz_heatmap, self.visualizer),
        }
        self.transform_map = {
            'none': functools.partial(self.__no_transform),
            'minmax': functools.partial(self.__minmax_normalization)
        }
        self.colorize_map = { "none": lambda x: x }
        self.colorize_map.update(COLORMAPS)

    @property
    def name(self) -> str:
        return self.env_name
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for k, t, tf, c in zip(self.images, self.types, self.transforms, self.colormaps):
            if not k:
                continue
            tensor = tensors[k]
            b, h, _, __ = tensor.size()            
            for i in range(h):
                key = k + "_{}".format(i)
                self.viz_map[t](
                    self.colorize_map[c](
                        self.transform_map[tf](
                            tensor[:, i, ...].unsqueeze(1)
                        )), key, key, self.name
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
        b, _, __, ___ = tensor.size() # assumes [B, C, H, W], i.e. 2d train
        heatmaps = torch.flip(tensor, dims=[2]).detach().cpu()
        for i in range(b):
            opts = (
            {
                'title': key + "_{}".format(i),
                'colormap': 'Viridis'
            })
            visdom.heatmap(heatmaps[i, :, :, :].squeeze(), opts=opts, win=win + str(i))

    @staticmethod #TODO: refactor these into a common module
    def __no_transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod #TODO: refactor these into a common module
    def __minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
        b, c, __, ___ = tensor.size()
        min_v = torch.min(tensor.view(b, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        max_v = torch.max(tensor.view(b, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        return (tensor - min_v) / (max_v - min_v)
    


