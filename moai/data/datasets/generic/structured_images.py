from moai.data.datasets.common import load_color_image
from moai.utils.arguments import ensure_string_list

import torch
import glob
import os
import typing
import logging

__all__ = ["StructuredImages"]

log = logging.getLogger(__name__)

class StructuredImages(torch.utils.data.Dataset):
    def __init__(self,
        root:           str='',
        **kwargs:       typing.Mapping[str, typing.Mapping[str, typing.Any]],
    ):
        self.key_to_list = {}
        self.key_to_xform = {}
        for k, m in kwargs.items():
            glob_list = ensure_string_list(m['glob'])
            files = []
            for g in glob_list:
                files += glob.glob(os.path.join(root, g))
            self.key_to_list[k] = list(map(lambda f: os.path.join(root, f), files))
            self.key_to_xform[k] = m['output_space']

    def __len__(self) -> int:
        return len(next(iter(self.key_to_list.values())))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        ret = { }
        for k, l in self.key_to_list.items():
            ret[k] = load_color_image(l[index], self.key_to_xform[k])
        return ret