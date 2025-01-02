import logging
import typing

import hydra.utils as hyu
import numpy as np
import omegaconf.omegaconf
import toolz
import torch

log = logging.getLogger(__name__)

__all__ = ["Windowed"]


def __merge_func__(x):
    # merge values from dict using toolz
    if isinstance(x, dict):
        return toolz.merge_with(__merge_func__, x)
    elif isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
        elif isinstance(x[0], np.ndarray):
            return torch.stack([torch.from_numpy(x) for x in x])
        else:
            if not isinstance(x[0], dict):
                return x[0]
            else:
                return toolz.merge_with(__merge_func__, x)


class Windowed(torch.utils.data.Dataset):
    r"""
    Dataset creation by getting sequential values from one dataset.

    Args:
        datasets (sequence): DictConfig of datasets to be concatenated
        window_size (int): Size of the window to be sampled
        stride (int): Stride >= 1 to be used for sampling
    """

    def __init__(
        self,
        datasets: omegaconf.DictConfig,
        window_size: int,
        stride: int = 1,
        internal_stride: int = 1,
        augmentation: omegaconf.DictConfig = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.internal_stride = internal_stride
        if augmentation is not None:
            self.dataset = hyu.instantiate(
                augmentation, hyu.instantiate(next(iter(datasets.values())))
            )
        else:
            from moai.data.augmentation import NoOp

            self.dataset = NoOp(hyu.instantiate(next(iter(datasets.values()))))
        if self.window_size <= 0:
            log.warning(f"Window size must be > 0. Using window size = 1.")
            self.window_size = 1
        if self.window_size > len(self.dataset):
            log.warning(
                f"Window size must not be bigger > than the dataset length. Using window size = {len(self.dataset)}."
            )
            self.window_size = len(self.dataset)
        if stride <= 0:
            log.warning(f"Stride must be > 0. Using stride = 1.")
            self.stride = 1
        if internal_stride <= 0:
            log.warning(f"Internal stride must be > 0. Using internal stride = 1.")
            self.internal_stride = 1
        log.info(
            f"Loaded {len(self)} windows of size {self.window_size} with stride {self.stride} from {len(self.dataset)} samples."
        )

    def __len__(self) -> int:
        size = 0
        for index in range(len(self.dataset)):
            last_frame_position = (
                index * self.stride + (self.window_size - 1) * self.internal_stride
                if self.window_size > 1
                else index * self.stride + self.internal_stride
            )
            if last_frame_position >= len(self.dataset):
                break
            size += 1
        return size

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        out = toolz.merge_with(
            __merge_func__,
            (
                [
                    self.dataset[i * self.internal_stride + index * self.stride]
                    for i in range(0, self.window_size)
                ]
                if self.window_size > 1
                else self.dataset[self.internal_stride + index * self.stride]
            ),
        )  # NOTE: Check if it is too slow

        return out
