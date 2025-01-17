import functools
import logging
import os
import typing
from collections import defaultdict

import ffmpegio
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from moai.utils.arguments import ensure_choices, ensure_path

__all__ = ["Video2d"]

log = logging.getLogger(__name__)

_internal_video_id_ = 0


def _create_video_writer(
    path: str, overwrite: bool, prefix: str, suffix: str, lossy: bool, fps: float = 60.0
):
    global _internal_video_id_
    sp_kwargs = (
        {"c:v": "libx264"}
        if lossy
        else {"c:v": "libx265", "qp": "0", "x265-params": "lossless=1"}
    )
    vid = ffmpegio.open(
        (
            os.path.join(path, f"{prefix}{_internal_video_id_}{suffix}.mp4")
            if os.path.isdir(path)
            else path
        ),
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
        # NOTE: forced lossless coding params and x265
        # sp_kwargs={"c:v": "libx265", "qp": "0", "x265-params": "lossless=1"},
        sp_kwargs=sp_kwargs,
        # TODO: make such  profiles selectable, e.g. lossless gray8, lossy rgb8, etc.
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
        prefix: str = "",
        suffix: str = "",
        lossy: bool = False,
    ):
        self.path = (
            ensure_path(log, "output folder", path) if os.path.isdir(path) else path
        )
        self.format = ensure_choices(
            log, "output format", extension, MultiviewVideo2d.__FORMATS__
        )
        self.overwrite = overwrite
        self.prefix = prefix
        self.suffix = suffix
        self.lossy = lossy

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
        fps: np.ndarray,
        batch_idx: typing.Optional[int] = None,
    ) -> None:
        if batch_idx == 0:
            self.video_writer_lambda = functools.partial(
                _create_video_writer,
                path=self.path,
                overwrite=self.overwrite,
                prefix=self.prefix,
                suffix=self.suffix,
                lossy=self.lossy,
                fps=float(np.ceil(fps.squeeze())),
            )
            self.video_writers = defaultdict(self.video_writer_lambda)
        for i, color in enumerate(images):
            img = color.transpose(1, 2, 0)
            img *= 255
            self.video_writers[i].write(img.astype(np.uint8))
