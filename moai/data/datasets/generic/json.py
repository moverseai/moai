from moai.data.datasets.common import load_json_file

import torch
import glob
import os
import typing
import logging
import toolz

__all__ = ["Json"]

log = logging.getLogger(__name__)


class Json(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "",
    ):
        self.files = glob.glob(os.path.join(root, "*.json"))
        log.info(f"Loaded {len(self)} .json files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return toolz.valmap(
            lambda t: t.detach() if isinstance(t, torch.Tensor) else t,
            load_json_file(self.files[index]),
        )
