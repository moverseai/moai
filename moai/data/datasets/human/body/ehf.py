from moai.utils.arguments import assert_path
from moai.data.datasets.common.image2d import load_color_image

import json
import numpy as np
import cv2
import glob
import torch
import os
import typing
import logging
import trimesh

log = logging.getLogger(__name__)

__all__ = ['EHF']

class EHF(torch.utils.data.Dataset):
    def __init__(self,
        root:               str,
        load_jpg:           bool=False,
        use_face_contour:   bool=False,
        # load_scans:     bool=False,
    ) -> None:
        super().__init__()
        assert_path(log, 'EHF root path', root)
        self.root = root
        # self.load_scans = load_scans        
        self.fits = glob.glob(os.path.join(root, 'fits', '*.ply'))
        if load_jpg:
            self.images = glob.glob(os.path.join(root, 'images', '*.jp*g'))
        else:
            self.images = glob.glob(os.path.join(root, 'png', '*.png'))
        self.focal_length = torch.Tensor([1498.22426237, 1498.22426237]).float()
        self.principal_point = torch.Tensor([790.263706, 578.90334]).float()
        self.camera_translation = torch.Tensor([-0.03609917,  0.43416458,  2.37101226]).float()
        self.camera_rotation = torch.Tensor([-2.98747896,  0.01172457, -0.05704687]).float()
        self.intrinsics = torch.Tensor([
            [1498.22426237, 0.0, 790.263706],
            [0.0, 1498.22426237, 578.90334],
            [0.0, 0.0, 1.0],
        ]).float()
        xform = np.eye(4, dtype=np.float32)
        xform[:3, :3] = cv2.Rodrigues(self.camera_rotation.numpy())[0]
        xform[3, :3] = self.camera_translation.numpy().copy()
        self.extrinsics = torch.from_numpy(xform)
        self.load_face_contour = use_face_contour
        
    def __len__(self) -> int:
        return len(self.fits)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        fit_filename = self.fits[index]
        img_filename = self.images[index]
        base, ext = os.path.splitext(os.path.basename(img_filename))
        image = load_color_image(img_filename)
        smplx = trimesh.load(fit_filename, process=False)
        keypoints, confidence = self._read_keypoints(os.path.join(
            self.root, 'keypoints', f"{base}_keypoints.json"
        ))
        data = {
            'color': image,
            'keypoints': keypoints,
            'confidence': confidence,
            'vertices': torch.from_numpy(
                np.dot(smplx.vertices, self.extrinsics[:3, :3].numpy().T) + self.camera_translation.numpy()
            ).float(),
            'faces': torch.from_numpy(smplx.faces).int(), #NOTE or long?
            'camera_rotation': self.camera_rotation,
            'camera_translation': self.camera_translation,
            'focal_length': self.focal_length,
            'principal_point': self.principal_point,
            'intrinsics': self.intrinsics,
            'extrinsics': self.extrinsics,
        }
        # if self.load_scans:
        #     scan = trimesh.load(fit_filename.replace('_align.ply', '_scan.obj'))
        #     data['scan_vertices'] = torch.from_numpy(scan.vertices).float()
        #     data['scan_faces'] = torch.from_numpy(scan.faces).int() #NOTE: or long?
        return data

    def _read_keypoints(self, filename: str) -> torch.Tensor:
        with open(filename) as keypoint_file:
            data = json.load(keypoint_file)
        person = data['people'][0]
        body = np.array(person['pose_keypoints_2d'], dtype=np.float32)
        body = body.reshape([-1, 3])
        if True: # load_hands:
            left_hand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            right_hand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            body = np.concatenate([body, left_hand, right_hand], axis=0)
        if True: # load_face:
            face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
            if self.load_face_contour:
                contour_keyps = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[:17, :]
            body = np.concatenate([body, face, contour_keyps], axis=0)            
        return torch.from_numpy(body[:, :2]), torch.from_numpy(body[:, 2:3])