from moai.data.datasets.common import load_txt_file

import torch
import glob
import os
import typing
import logging

__all__ = ["Txt"]

log = logging.getLogger(__name__)

class Txt(torch.utils.data.Dataset):
    def __init__(self,
        root:           str='',
    ):
        self.files = glob.glob(os.path.join(root, '*.txt'))
        log.info(f"Loaded {len(self)} .txt files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return load_txt_file(self.files[index])