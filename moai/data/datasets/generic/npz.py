import glob
import logging
import os
import typing

import toolz
import torch

from moai.data.datasets.common import load_npz_file

__all__ = ["Npz", "StandaloneNpz", "RepeatedNpz"]

log = logging.getLogger(__name__)


class Npz(
    torch.utils.data.Dataset
):  # TODO add the loading different npz and combining them in the same dict case
    def __init__(
        self,
        root: str = "",
        extra_key: str = None,
    ):
        self.files = glob.glob(os.path.join(root, "*.npz"))
        self.extra_key = extra_key
        log.info(f"Loaded {len(self)} .npz files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return (
            load_npz_file(self.files[index])
            if self.extra_key is None
            else {self.extra_key: load_npz_file(self.files[index])}
        )


class StandaloneNpz(
    torch.utils.data.Dataset
):  # TODO add the loading different npz and combining them in the same dict case
    def __init__(
        self,
        filename: str = "",
    ):
        self.file = load_npz_file(filename)
        log.info(f"Loaded an .npz file producing [{list(self.file.keys())}].")

    def __len__(self) -> int:
        return len(self.file[toolz.first(self.file)])

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return toolz.valmap(lambda t: t[index], self.file)


class RepeatedNpz(
    torch.utils.data.Dataset
):  # TODO add the loading different npz and combining them in the same dict case
    def __init__(
        self,
        length: int,
        filename: str = "",
    ):
        self.file = load_npz_file(filename)
        self.length = length
        log.info(f"Loaded an .npz file producing [{list(self.file.keys())}].")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return self.file
