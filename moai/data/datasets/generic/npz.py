from moai.data.datasets.common import load_npz_file

import torch
import glob
import os
import typing
import logging

__all__ = ["Npz"]

log = logging.getLogger(__name__)

class Npz(torch.utils.data.Dataset):            #TODO add the loading different npz and combining them in the same dict case
    def __init__(self,
        root:           str='',
    ):
        self.files = glob.glob(os.path.join(root, '*.npz'))
        log.info(f"Loaded {len(self)} .npz files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return load_npz_file(self.files[index])