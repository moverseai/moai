import functools
import logging
import os
import typing
from collections import defaultdict
from collections.abc import Callable

import ffmpegio
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

import moai.utils.color.colorize as mic
from moai.utils.arguments import ensure_choices, ensure_path

__all__ = ["Video2d"]

log = logging.getLogger(__name__)

_internal_video_id_ = 0


def _create_video_writer(path: str, overwrite: bool, fps: float = 60.0):
    global _internal_video_id_
    vid = ffmpegio.open(
        f"{path}/{_internal_video_id_}.mp4",
        "wv",
        fps,
        overwrite=overwrite,
        # hwaccel='cuvid',
        # hwaccel='cuda',
        # crf=0,
        # sp_kwargs={
        #     # "overwrite": overwrite,
        #     'hwaccel': 'cuda',
        #     'c:v': 'h264_cuda',
        #     # 'hwaccel': 'cuvid',
        #     # 'c:v': 'h264_cuvid',
        # },
        sp_kwargs={"c:v": "libx265", "qp": "0", "x265-params": "lossless=1"},
    )
    _internal_video_id_ += 1
    return vid


class MultiviewVideo2d(
    Callback,
    typing.Callable[
        [typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]],
        None,
    ],
):
    __FORMATS__ = ["mp4"]

    def __init__(
        self,
        path: str,
        extension: str = "mp4",
        overwrite: bool = False,
    ):
        self.path = ensure_path(log, "output folder", path)
        self.format = ensure_choices(
            log, "output format", extension, MultiviewVideo2d.__FORMATS__
        )
        self.overwrite = overwrite

    def on_predict_epoch_end(self, *args):
        self.close()

    def on_test_epoch_end(self, *args):
        self.close()

    def on_train_epoch_end(self, *args):
        self.close()

    def close(self):
        log.info(f"Writing data to {self.path}")
        for _, v in self.video_writers.items():
            if not v.closed:
                v.close()
        self.video_writers = defaultdict(self.video_writer_lambda)

    def __call__(
        self,
        images: np.ndarray,  # TODO: add fps
        batch_idx: typing.Optional[int] = None,
    ) -> None:
        if batch_idx == 0:
            self.video_writer_lambda = functools.partial(
                _create_video_writer, path=self.path, overwrite=self.overwrite
            )
            self.video_writers = defaultdict(self.video_writer_lambda)
        for i, color in enumerate(images):
            img = color.transpose(1, 2, 0)
            img *= 255
            self.video_writers[i].write(img.astype(np.uint8))
