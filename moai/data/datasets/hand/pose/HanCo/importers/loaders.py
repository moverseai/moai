import torch
import json
import cv2
import numpy as np
import typing

def load_color(filenames: str, idx: int, **kwargs) -> torch.Tensor:
    img = torch.from_numpy(
            cv2.imread(filenames['rgb_filenames'][idx]).transpose(2, 0, 1)
        ).flip(dims=[0])
    return {'color': img / 255.0 }

def load_fg_mask(filenames: str, idx: int, **kwargs) -> torch.Tensor:
    img = torch.from_numpy(
            cv2.imread(filenames['mask_fg_filenames'][idx]).transpose(2, 0, 1)
        ).flip(dims=[0])
    return {'mask_fg': img[0:1, ...] / 255.0 }
    
def load_hand_mask(filenames: str, idx: int, **kwargs) -> torch.Tensor:
    img = torch.from_numpy(
            cv2.imread(filenames['mask_hand_filenames'][idx]).transpose(2, 0, 1)
        ).flip(dims=[0])
    return {'mask_hand': img[0:1, ...] / 255.0 }

def load_keypoints3D(filenames: str, idx: int, **kwargs) -> torch.Tensor:
    if idx > len(filenames['calib_filenames']):
        idx = (idx // 8) + (idx % 8)
    with open(filenames['keypoints_filenames'][idx], 'r') as fi:
        keypoints_xyz = json.load(fi)
    return {'keypoints3D': torch.from_numpy(np.array(keypoints_xyz)).transpose(1,0)}

def load_keypoints2D(filenames: str, idx: int, **kwargs) -> torch.Tensor:
    viewpoint = filenames['rgb_filenames'][idx].split('\\')[-2].split('cam')[-1]
    if idx > len(filenames['calib_filenames']):
        idx = (idx // 8) + (idx % 8)
    with open(filenames['keypoints_filenames'][idx], 'r') as fi:
        keypoints_xyz = json.load(fi)
    with open(filenames['calib_filenames'][idx], 'r') as fi:
        calib_params = json.load(fi)
    keypoints2D = projectPoints(keypoints_xyz, calib_params['K'][int(viewpoint)])
    
    return {'keypoints2D': torch.from_numpy(keypoints2D).transpose(1,0)}

def load_mano(filenames: str, idx: int, **kwargs) -> torch.Tensor:
    with open(filenames['mano_filenames'][idx], 'r') as fi:
        mano_params = json.load(fi)
    poses, betas, global_t = split_theta(mano_params)
    mano = {}
    mano['betas'] = torch.from_numpy(np.array(betas))         # 10 shape parameters
    mano['poses'] = torch.from_numpy(np.array(poses))         # 45 rotation parameters (for 15 joints) + 3 global rotation parameters
    mano['global_t'] = torch.from_numpy(np.array(global_t))   # 3 global translation parameters
    return {'mano': mano}

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def split_theta(theta):
    shapes = theta[:10]
    poses = theta[10:58]
    global_t = theta[58:61]
    return poses, shapes, global_t
