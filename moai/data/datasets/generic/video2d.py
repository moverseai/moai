import glob
import logging
import os
import typing

import toolz
import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from moai.utils.arguments import ensure_path

__all__ = ["Video2d"]

log = logging.getLogger(__name__)


class Video2d(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
    ):
        path = ensure_path(log, "video", path)
        self.video = EncodedVideo.from_path(
            path,
            decode_audio=False,
            decoder="decord",
        )
        log.info(f"Loaded a video file with {len(self)} frames from {path}.")
        self.start_end = (0, len(self))

    def __len__(self) -> int:
        return int(self.video.duration * self.video._fps)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        start_clip = (index + self.start_end[0]) / self.video._fps
        end_clip = start_clip + 1 / self.video._fps
        return {
            "frame": self.video.get_clip(start_clip - 1e-7, end_clip - 1e-7)[
                "video"
            ].squeeze(1)
            / 255.0,
            "framerate": torch.scalar_tensor(self.video._fps),
        }
