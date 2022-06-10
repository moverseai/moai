from moai.data.datasets.common import load_color_image
from moai.data.datasets.common.image2d import load_depth_image, load_mask_image, load_gray_image
from moai.utils.arguments import ensure_string_list

import torch
import glob
import os
import typing
import logging
import toolz

__all__ = ["StructuredImages"]

log = logging.getLogger(__name__)

_LOADERS_ = {
    'color':        load_color_image,
    'image':        load_color_image,
    'gray':         load_gray_image,
    'depth':        load_depth_image,
    'range':        load_depth_image,
    'mask':         load_mask_image,
    'silhouette':   load_mask_image,
}

class StructuredImages(torch.utils.data.Dataset):
    def __init__(self,
        root:           str='',
        **kwargs:       typing.Mapping[str, typing.Mapping[str, typing.Any]],
    ):
        self.key_to_list = {}
        self.key_to_params = {}
        self.key_to_loader = {}
        for k, m in kwargs.items():
            glob_list = ensure_string_list(m['glob'])
            files = []
            for g in glob_list:
                files += glob.glob(os.path.join(root, g))
            self.key_to_list[k] = list(map(lambda f: os.path.join(root, f), files))
            self.key_to_params[k] = toolz.dissoc(m, 'type', 'glob')
            self.key_to_loader[k] = _LOADERS_[m.type or k] #m['type']

    def __len__(self) -> int:
        return len(next(iter(self.key_to_list.values())))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        ret = { }
        for k, l in self.key_to_list.items():
            ret[k] = self.key_to_loader[k](l[index], **self.key_to_params[k])
        return ret