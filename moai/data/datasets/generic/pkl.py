import glob
import logging
import os
import typing

import toolz
import torch

from moai.data.datasets.common import load_pkl_file

__all__ = ["Pkl"]

log = logging.getLogger(__name__)


class Pkl(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "",
    ):
        self.files = glob.glob(os.path.join(root, "*.pkl"))
        log.info(f"Loaded {len(self)} .pkl files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return toolz.valmap(
            lambda t: t.detach() if isinstance(t, torch.Tensor) else t,
            load_pkl_file(self.files[index]),
        )
