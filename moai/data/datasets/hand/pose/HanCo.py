from asyncio.windows_events import NULL
from moai.data.datasets.common import load_color_image
from moai.data.datasets.common.image2d import load_mask_image

import os
import cv2
import sys
import torch
import glob
import json
import typing
import logging
import numpy as np
from collections import namedtuple


__all__ = [
    "HanCoLoader",
]

logger = logging.getLogger(__name__)

# HanCoParams = namedtuple('HanCoParams', [
#     'load_rgb',
#     'load_mano',
#     'load_keypoints',
# ])

class HanCo(torch.utils.data.Dataset):
    def __init__(self,
        root:               str,
        data_types:         str,
    ):
        super(HanCo, self).__init__()
        if 'color' in data_types:
            self.rgb_image_filenames = glob.glob(os.path.join(root,'rgb','**','*.jpg'), recursive=True)
        else:
            sys.exit("No 'color' data type")
        if 'mask_hand' in data_types:
            self.mask_hand_image_filenames = glob.glob(os.path.join(root,'mask_hand','**','*.jpg'), recursive=True)
        else:
            sys.exit("No 'mask_hand' data type")
        if 'mask_fg' in data_types:
            self.mask_fg_image_filenames = glob.glob(os.path.join(root,'mask_fg','**','*.jpg'), recursive=True)
        else:
            sys.exit("No 'mask_fg' data type")

        #keypoints
        K_all_cams = glob.glob(os.path.join(root,'calib','**','*.json'), recursive=True)
        K = [v for v in K_all_cams for _ in range(8)]
        self.K_filenames = {}
        for i, val in enumerate(K):
            self.K_filenames[i] = {}
            self.K_filenames[i]['K'] = val
            self.K_filenames[i]['pos'] = i % 8     
        keypoints_filenames_all_cams =  glob.glob(os.path.join(root,'xyz','**','*.json'), recursive=True)
        self.keypoints_filenames = [v for v in keypoints_filenames_all_cams for _ in range(8)]

        #mano
        mano = glob.glob(os.path.join(root,'shape','*.json'))
        self.mano_filenames = [v for v in mano for _ in range(8)]

    def __len__(self) -> int:
        return len(self.rgb_image_filenames)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        rgb_img_filename = self.rgb_image_filenames[index]
        img = load_color_image(rgb_img_filename)
        
        mask_hand_img_filename = self.mask_hand_image_filenames[index]
        mask_fg_img_filename = self.mask_fg_image_filenames[index]
        mask_hand = load_mask_image(mask_hand_img_filename)
        mask_fg = load_mask_image(mask_fg_img_filename)

        keypoints_xyz = self._json_load(self.keypoints_filenames[index])
        cam_idx = self.K_filenames[index]['pos']
        K_list = self._json_load(self.K_filenames[index]['K'])
        K = {}
        K['K'] = K_list['K'][cam_idx]
        K['M'] = K_list['M'][cam_idx]
        K, keypoints_xyz = [np.array(x) for x in [K['K'], keypoints_xyz]]
        uv = self._projectPoints(keypoints_xyz, K)

        mano_values = self._json_load(self.mano_filenames[index])
        poses, betas, global_t = self._split_theta(mano_values)
        mano = {}
        mano['poses'] = poses
        mano['betas'] = betas
        mano['global_t'] = global_t

        # img = self._draw_hand(img.permute(1,2,0).numpy(), uv, order='uv', img_order='rgb')
        
        return {
            'color': img,
            'mask_hand': mask_hand,
            'mask_fg': mask_fg,
            'keypoints': keypoints_xyz,
            'uv': uv,
            'mano': mano
        }

    def _projectPoints(self, xyz, K):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]

    def _split_theta(self, theta):
        poses = theta['poses'][0]
        shapes = theta['shapes'][0]
        global_t = theta['global_t'][0][0]
        return poses, shapes, global_t

    def _json_load(self, p):
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d

    def _draw_hand(self, image, coords_hw, vis=None, color_fixed=None, linewidth=3, order='hw', img_order='rgb',
              draw_kp=True, kp_style=None):
        """ Inpaints a hand stick figure into a matplotlib figure. """
        if kp_style is None:
            kp_style = (5, 3)

        image = np.squeeze(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        s = image.shape
        assert len(s) == 3, "This only works for single images."

        convert_to_uint8 = False
        if s[2] == 1:
            # grayscale case
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
            image = np.tile(image, [1, 1, 3])
            pass
        elif s[2] == 3:
            # RGB case
            if image.dtype == np.uint8:
                convert_to_uint8 = True
                image = image.astype('float32') / 255.0
            elif image.dtype == np.float32:
                # convert to gray image
                image = np.mean(image, axis=2)
                image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
                image = np.expand_dims(image, 2)
                image = np.tile(image, [1, 1, 3])
        else:
            assert 0, "Unknown image dimensions."

        if order == 'uv':
            coords_hw = coords_hw[:, ::-1]

        colors = np.array([[0.4, 0.4, 0.4],
                        [0.4, 0.0, 0.0],
                        [0.6, 0.0, 0.0],
                        [0.8, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.4, 0.4, 0.0],
                        [0.6, 0.6, 0.0],
                        [0.8, 0.8, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.4, 0.2],
                        [0.0, 0.6, 0.3],
                        [0.0, 0.8, 0.4],
                        [0.0, 1.0, 0.5],
                        [0.0, 0.2, 0.4],
                        [0.0, 0.3, 0.6],
                        [0.0, 0.4, 0.8],
                        [0.0, 0.5, 1.0],
                        [0.4, 0.0, 0.4],
                        [0.6, 0.0, 0.6],
                        [0.7, 0.0, 0.8],
                        [1.0, 0.0, 1.0]])

        if img_order == 'rgb':
            colors = colors[:, ::-1]

        # define connections and colors of the bones
        bones = [((0, 1), colors[1, :]),
                ((1, 2), colors[2, :]),
                ((2, 3), colors[3, :]),
                ((3, 4), colors[4, :]),

                ((0, 5), colors[5, :]),
                ((5, 6), colors[6, :]),
                ((6, 7), colors[7, :]),
                ((7, 8), colors[8, :]),

                ((0, 9), colors[9, :]),
                ((9, 10), colors[10, :]),
                ((10, 11), colors[11, :]),
                ((11, 12), colors[12, :]),

                ((0, 13), colors[13, :]),
                ((13, 14), colors[14, :]),
                ((14, 15), colors[15, :]),
                ((15, 16), colors[16, :]),

                ((0, 17), colors[17, :]),
                ((17, 18), colors[18, :]),
                ((18, 19), colors[19, :]),
                ((19, 20), colors[20, :])]

        color_map = {'k': np.array([0.0, 0.0, 0.0]),
                    'w': np.array([1.0, 1.0, 1.0]),
                    'b': np.array([0.0, 0.0, 1.0]),
                    'g': np.array([0.0, 1.0, 0.0]),
                    'r': np.array([1.0, 0.0, 0.0]),
                    'm': np.array([1.0, 1.0, 0.0]),
                    'c': np.array([0.0, 1.0, 1.0])}

        if vis is None:
            vis = np.ones_like(coords_hw[:, 0]) == 1.0

        for connection, color in bones:
            if (vis[connection[0]] == False) or (vis[connection[1]] == False):
                continue

            coord1 = coords_hw[connection[0], :].astype(np.int32)
            coord2 = coords_hw[connection[1], :].astype(np.int32)

            if (coord1[0] < 1) or (coord1[0] >= s[0]) or (coord1[1] < 1) or (coord1[1] >= s[1]):
                continue
            if (coord2[0] < 1) or (coord2[0] >= s[0]) or (coord2[1] < 1) or (coord2[1] >= s[1]):
                continue

            if color_fixed is None:
                cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), color, thickness=linewidth)
            else:
                c = color_map.get(color_fixed, np.array([1.0, 1.0, 1.0]))
                cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), c, thickness=linewidth)

        if draw_kp:
            coords_hw = coords_hw.astype(np.int32)
            for i in range(21):
                if vis[i]:
                    # cv2.circle(img, center, radius, color, thickness)
                    image = cv2.circle(image, (coords_hw[i, 1], coords_hw[i, 0]),
                                    radius=kp_style[0], color=colors[i, :], thickness=kp_style[1])

        if convert_to_uint8:
            image = (image * 255).astype('uint8')

        return image