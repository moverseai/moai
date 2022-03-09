from moai.utils.arguments import assert_path
from moai.data.datasets.common.image2d import (
    load_color_image,
    load_binary_image,
)

import numpy as np
import glob
import torch
import os
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['SSP3D']

#NOTE: Human3.6M body structure: {
    # 0: 'Pelvis',
    # 1: 'RHip',
    # 2: 'RKnee',
    # 3: 'RAnkle',
    # 4: 'LHip',
    # 5: 'LKnee',
    # 6: 'LAnkle',
    # 7: 'Spine1',
    # 8: 'Neck',
    # 9: 'Head',
    # 10: 'Site',
    # 11: 'LShoulder',
    # 12: 'LElbow',
    # 13: 'LWrist',
    # 14: 'RShoulder',
    # 15: 'RElbow',
    # 16: 'RWrist,
# };

#NOTE: COCO17 body structure: {
    # 0: 'Nose',
    # 1: 'REye',
    # 2: 'LEye',
    # 3: 'REar',
    # 4: 'LEar',
    # 5: 'RShoulder',
    # 6: 'LShoulder',
    # 7: 'RElbow',
    # 8: 'LElbow',
    # 9: 'RWrist',
    # 10: 'LWrist',
    # 11: 'RHip',
    # 12: 'LHip',
    # 13: 'RKnee',
    # 14: 'LKnee',
    # 15: 'RAnkle',
    # 16: 'LAnkle,
# };

class SSP3D(torch.utils.data.Dataset):
    def __init__(self,
        root:           str,
        betas:          int=10,
    ) -> None:
        super().__init__()
        assert_path(log, 'EHF root path', root)
        labels = np.load(os.path.join(root, 'labels.npz'))
        self.root = root
        # self.silhouettes = glob.glob(os.path.join(root, 'silhouettes', '*.png'))
        # self.images = glob.glob(os.path.join(root, 'images', '*.png'))
        self.betas = betas
        for k, v in labels.items():
            setattr(self, k, torch.from_numpy(v) if np.issubdtype(v.dtype, np.number) else v)

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return {
            'color': load_color_image(os.path.join(self.root, 'images', self.fnames[index])),
            'silhouette': load_binary_image(os.path.join(self.root, 'silhouettes', self.fnames[index])),
            'shape': torch.cat([self.shapes[index], torch.zeros(self.betas-10)], dim=-1) if self.betas > 10 else self.shapes[index],
            'global_orientation': self.poses[index][:3],
            'pose': self.poses[index][3:],
            'keypoints': self.joints2D[index][:, :2],
            'confidence': self.joints2D[index][:, 2:3],
            'camera_translation': self.cam_trans[index],
        }