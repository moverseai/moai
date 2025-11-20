import logging
import typing
from collections import defaultdict

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


class DynamicWindowed(torch.utils.data.Dataset):
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
        self.window_args = defaultdict(dict)
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
        try:
            actors = self.dataset.inner.num_actors
        except AttributeError:
            actors = 1
        # TODO: if dataset maximum size if bigger than window size but only for a small number of frames
        # we should take the whole dataset as one window
        # window size should be adapted and scaled down depeding on the number of actors
        if actors > 4:
            self.window_size = max(1, self.window_size // actors)
            self.stride = self.window_size - 1
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
        dataset_length = len(self.dataset)

        if dataset_length <= self.window_size:
            # should return 1 window with all the data
            self.window_size = dataset_length
            self.stride = self.window_size - 1
            return 1

        # Calculate how many full windows we can have
        full_windows = (dataset_length - 1) // self.stride

        # Calculate remaining frames after full windows
        last_frame_of_full_windows = (full_windows - 1) * self.stride + self.window_size
        remaining_frames = dataset_length - last_frame_of_full_windows

        # If we have remaining frames, add one more window
        if remaining_frames > 0:
            # Adjust window size and stride for the last window
            self.last_window_size = remaining_frames
            self.last_window_stride = remaining_frames - 1
            return full_windows + 1

        return full_windows

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        if index == len(self) - 1 and hasattr(self, "last_window_size"):
            # Use adjusted window size and stride for last window
            current_window_size = self.last_window_size
            current_stride = self.last_window_stride
        else:
            current_window_size = self.window_size
            current_stride = self.stride
        out = toolz.merge_with(
            __merge_func__,
            (
                [
                    self.dataset[
                        # (
                        i * self.internal_stride
                        + index * self.stride
                        # if i == 0
                        # else i * self.internal_stride + index * current_stride
                        # )
                    ]
                    for i in range(0, current_window_size)
                ]
                if current_window_size > 1
                else self.dataset[self.internal_stride + index * current_stride]
            ),
        )

        return out
