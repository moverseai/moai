from moai.data.datasets.common import load_npz_files
from moai.utils.arguments import ensure_string_list

import torch
import glob
import os
import typing
import logging
import toolz

__all__ = ["Npz"]

log = logging.getLogger(__name__)

class Npz(torch.utils.data.Dataset):            #TODO add the loading different npz and combining them in the same dict case
    def __init__(self,
        root:           str='',
        **kwargs:       typing.Mapping[str, typing.Mapping[str, typing.Any]],
    ):
        self.files = glob.glob(os.path.join(root,'*.npz'))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        ret = load_npz_files(self.files[index])
        return ret