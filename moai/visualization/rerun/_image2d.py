import logging
import typing
import functools
import torch
import numpy as np
import math
import toolz
import cv2

from moai.utils.arguments import assert_numeric

log = logging.getLogger(__name__)

from collections.abc import Callable

try:
    import rerun as rr
except:
    log.error(f"Please `pip install rerun-sdk` to use ReRun visualisation module.")


# create a callable image2d visualizer
class Image2d(Callable):
    def __init__(
        self,
        name: str,  # Path to the image in the space hierarchy,
        image: typing.Union[str, typing.Sequence[str]],
        type: typing.Union[str, typing.Sequence[str]],
        colormap: typing.Union[str, typing.Sequence[str]],
        transform: typing.Union[str, typing.Sequence[str]],
        path_to_hierarchy: typing.Sequence[str],
        # rotations: typing.Sequence[str],
        batch_percentage: float = 1.0,
    ):
        rr.init(name)
        rr.spawn()
        self.name = name
        self.keys = [image] if isinstance(image, str) else list(image)
        self.path_to_hierarchy = path_to_hierarchy
        self.colormaps = [colormap] if isinstance(colormap, str) else list(colormap)
        self.batch_percentage = batch_percentage
        assert_numeric(log, 'batch percentage', batch_percentage, 0.0, 1.0)
        # self.images = [image] if isinstance(image, str) else list(image)
        self.transforms = [transform] if isinstance(transform, str) else list(transform)
        # self.rotations = [rotations] if isinstance(rotations, str) else list(rotations)
        self.types = [type] if isinstance(type, str) else list(type)
        self.viz_map = {
            "color": self._viz_color,
        }
        self.colorize_map = {"none": lambda x: x}
        self.access = lambda td, k: toolz.get_in(k.split('.'), td)
        self.transform_map = {
            "none": functools.partial(self._passthrough),
            # "minmax": functools.partial(self._minmax_normalization),
            # "ndc": functools.partial(self._ndc_to_one),
            # "dataset": functools.partial(self._dataset_normalization),
        }
        self.step = 0

    @staticmethod  # TODO: refactor these into a common module
    def _passthrough(
        tensors: typing.Dict[str, torch.Tensor],
        key: str,
        count: int,
    ) -> torch.Tensor:
        return tensors  if isinstance(tensors,torch.Tensor) else \
            tensors[key][:count] if key in tensors else None

    def __call__(
        self, tensors: typing.Dict[str, torch.Tensor], step: typing.Optional[int] = None
    ) -> None:
        for k, t, tf, c, path in zip(self.keys, self.types, self.transforms, self.colormaps, self.path_to_hierarchy):
            self.viz_map[t](
                self.colorize_map[c](
                    self.transform_map[tf](
                        self.access(tensors, k) if '.' in k else tensors,
                        k,
                        int(
                            math.ceil(
                                self.batch_percentage * self.access(tensors, k).shape[0]
                            )
                        ),
                    )
                ),
                k,
                self.step,
                path,
            )
        self.step += 1

    @staticmethod
    def _viz_color(
        array: np.ndarray,
        key: str,
        step: int,
        env: str,
    ) -> None:
         for i, img in enumerate(array):
            rr.set_time_sequence("frame_nr", step)
            try:
                cv2_img = img.permute(1,2,0).detach().cpu().numpy()
            except:
                cv2_img = img.detach().cpu().numpy()
            flipped = cv2.flip(cv2_img, 0)

            rr.log_image(
                env,
                # image=img.detach().cpu().numpy(),
                # image=img.permute(2,1,0).detach().cpu().numpy(),
                # image=img.permute(1,2,0).detach().cpu().numpy(),
                image=flipped,
            )
