from moai.engine.modules.clearml import _get_logger
from moai.utils.color.colorize import COLORMAPS


import torch
import numpy as np
import clearml
import visdom
import functools
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Feature2d"]

class Feature2d(object):
    def __init__(self,
        image:            typing.Union[str, typing.Sequence[str]],
        type:             typing.Union[str, typing.Sequence[str]],
        colormap:         typing.Union[str, typing.Sequence[str]],
        transform:        typing.Union[str, typing.Sequence[str]],
        batch_percentage: float=1.0,        
        max_history:      int=50,
    ):
        #self.logger = _get_logger(project_name, task_name, uri, tags)
        self.logger = _get_logger()
        self.images = [image] if isinstance(image, str) else list(image)
        self.types = [type] if isinstance(type, str) else list(type)
        self.transforms = [transform] if isinstance(transform, str) else list(transform)
        self.colormaps = [colormap] if isinstance(colormap, str) is str else list(colormap)
        self.viz_map = {
            'color': functools.partial(self._viz_color, self.logger, max_history=max_history),
            # 'heatmap': functools.partial(self.__viz_heatmap, self.visualizer),
        }
        self.transform_map = {
            'none': functools.partial(self.__no_transform),
            'minmax': functools.partial(self.__minmax_normalization)
        }
        self.colorize_map = { "none": lambda x: x }
        self.colorize_map.update(COLORMAPS)
        #self.env_name = project_name

    # @property
    # def name(self) -> str:
    #     return self.env_name
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None
    ) -> None:
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
                        )), key, step, k #self.name
                )

    @staticmethod
    def _viz_color(
        logger:         clearml.Logger,
        array:          np.ndarray,
        key:            str,
        step:           int,
        env:            str,
        max_history:    int=50,
    ) -> None:
        for i, img in enumerate(array):
            logger.report_image(
                title=env, series=f"{key}_{i}", iteration=step, 
                image=img.transpose(1, 2, 0), max_image_history=max_history
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
    
    def _passthrough(
        tensors:    typing.Dict[str, torch.Tensor],
        key:        str,
        count:      int,
    ) -> torch.Tensor:
        return tensors[key][:count] if key in tensors else None

    @staticmethod #TODO: refactor these into a common module
    def __minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
        b, c, __, ___ = tensor.size()
        min_v = torch.min(tensor.view(b, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        max_v = torch.max(tensor.view(b, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        return (tensor - min_v) / (max_v - min_v)
    


