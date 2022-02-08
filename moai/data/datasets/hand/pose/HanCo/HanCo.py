from asyncio.windows_events import NULL

import os
import sys
import torch
import glob
import typing
import logging
import functools
import toolz
import json
from collections import namedtuple

import moai.data.datasets.hand.pose.HanCo.importers as importers

__all__ = [
    "HanCoLoader",
]

logger = logging.getLogger(__name__)

class HanCo(torch.utils.data.Dataset):
    __LOADERS__ = {
        'color': importers.loaders.load_color,
        'mask_fg': importers.loaders.load_fg_mask,
        'mask_hand': importers.loaders.load_hand_mask,
        'keypoints3D': importers.loaders.load_keypoints3D,
        'keypoints2D': importers.loaders.load_keypoints2D,
        'mano': importers.loaders.load_mano,
    }

    def __init__(self,
        root:               str,
        data_types:         str,
    ):
        super(HanCo, self).__init__()
        self.splits = ['train', 'val']
        self.filenames = self._get_filenames(root, self.splits)
        self.loaders = [
            functools.partial(
                HanCo.__LOADERS__[type]
            ) for type in data_types
        ]

    def _get_filenames(self,
        root: str,
        splits: typing.List[str]
    ) -> typing.Dict[str, typing.List[str]]:
        if not os.path.exists(root):
            logger.error((
                f"Invalid HanCo root folder ({root}), "
            ))
            sys.exit(2)

        base_filenames = glob.glob(os.path.join(root, 'shape', '**', '*.json'))

        rgb_base = [filename.replace('shape', 'rgb') for filename in base_filenames]
        rgb_base = [filename.replace('json', 'jpg') for filename in rgb_base]
        rgb_filenames = []
        for filename in rgb_base: 
            last = filename.split('\\')[-1]
            for viewpoint in range(8):
                new_name = filename.replace(last, os.path.join('cam' + str(viewpoint), last))
                rgb_filenames.append(new_name)

        shape_filenames = [filename.replace('rgb', 'shape') for filename in rgb_filenames] #this referes to the per image shape
        shape_filenames = [filename.replace('jpg', 'json') for filename in shape_filenames]
        mask_fg_filenames = [filename.replace('rgb', 'mask_fg') for filename in rgb_filenames]
        mask_hand_filenames = [filename.replace('rgb', 'mask_hand') for filename in rgb_filenames]
        xyz_filenames = [filename.replace('shape', 'xyz') for filename in base_filenames]
        calib_filenames = [filename.replace('shape', 'calib') for filename in base_filenames]

        return {
            'rgb_filenames': rgb_filenames,
            'mask_hand_filenames': mask_hand_filenames,
            'mask_fg_filenames': mask_fg_filenames,
            'mano_filenames': shape_filenames,
            'keypoints_filenames': xyz_filenames,
            'calib_filenames': calib_filenames
        }

    def __len__(self) -> int:
        return len(self.filenames['rgb_filenames'])

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        out = { }
        for loader in self.loaders:
            out = toolz.merge(out, loader(self.filenames, index))
        return out