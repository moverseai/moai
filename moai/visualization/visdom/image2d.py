from moai.visualization.visdom.base import Base
from moai.utils.color.colorize import get_colormap, COLORMAPS
from moai.utils.arguments import assert_numeric

import torch
import visdom
import functools
import typing
import logging
import numpy as np
import math

log = logging.getLogger(__name__)

__all__ = ["Image2d"]

class Image2d(Base):
    def __init__(self,
        image:             typing.Union[str, typing.Sequence[str]],
        type:              typing.Union[str, typing.Sequence[str]],
        colormap:          typing.Union[str, typing.Sequence[str]],
        transform:         typing.Union[str, typing.Sequence[str]],
        batch_percentage:   float=1.0,
        name:               str="default",
        ip:                 str="http://localhost",
        port:               int=8097,   
    ):
        super(Image2d, self).__init__(name, ip, port)
        self.keys = [image] if isinstance(image, str) else list(image)
        self.types = [type] if isinstance(type, str) else list(type)
        self.transforms = [transform] if isinstance(transform, str) else list(transform)
        self.colormaps = [colormap] if isinstance(colormap, str) else list(colormap)
        self.batch_percentage = batch_percentage
        assert_numeric(log, 'batch percentage', batch_percentage, 0.0, 1.0)
        self.viz_map = {
            'color': functools.partial(self._viz_color, self.visualizer),
            'heatmap': functools.partial(self._viz_heatmap, self.visualizer),
        }
        self.transform_map = {
            'none': functools.partial(self._passthrough),            
            'minmax': functools.partial(self._minmax_normalization),
            'ndc': functools.partial(self._ndc_to_one),
            'dataset': functools.partial(self._dataset_normalization),
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
                        tensors, k, int(math.ceil(self.batch_percentage * tensors[k].shape[0])),
                        # tensors[k][:int(self.batch_percentage * tensors[k].shape[0])]
                    )
                ), k, k, self.name
            )

    @staticmethod
    def _viz_color(
        visdom: visdom.Visdom,
        array: np.array,
        key: str,
        win: str,
        env: str
    ) -> None:
        visdom.images(
            array,
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        )

    @staticmethod
    def _viz_heatmap(
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
    def _passthrough(
        tensors:    typing.Dict[str, torch.Tensor],
        key:        str,
        count:      int,
    ) -> torch.Tensor:
        return tensors[key][:count] if key in tensors else None

    @staticmethod #TODO: refactor these into a common module
    def _minmax_normalization(
        tensors:    typing.Dict[str, torch.Tensor],
        key:        str,
        count:      int,
    ) -> torch.Tensor:
        tensor = Image2d._passthrough(tensors, key, count)
        b, _, __, ___ = tensor.size()
        min_v = torch.min(tensor.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        max_v = torch.max(tensor.view(b, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        return (tensor - min_v) / (max_v - min_v)

    @staticmethod #TODO: refactor these into a common module
    def _ndc_to_one(
        tensors:    typing.Dict[str, torch.Tensor],
        key:        str,
        count:      int,
    ) -> torch.Tensor:
        tensor = Image2d._passthrough(tensors, key, count)
        return torch.addcmul(
            torch.scalar_tensor(0.5).to(tensor), 
            tensor,
            torch.scalar_tensor(0.5).to(tensor)
        )

    @staticmethod #TODO: refactor these into a common module
    def _dataset_normalization(
        tensors:    typing.Dict[str, torch.Tensor],
        key:        str,
        count:      int,
    ) -> torch.Tensor:
        tensor = Image2d._passthrough(tensors, key, count)
        mean = Image2d._passthrough(tensors, 'dataset_mean', count)
        if mean is None:
            mean = torch.scalar_tensor(0.0).to(tensor)
        std = Image2d._passthrough(tensors, 'dataset_std', count)
        if std is None:
            std = torch.scalar_tensor(1.0).to(tensor)
        return torch.addcmul(
            mean, 
            tensor,
            std
        )