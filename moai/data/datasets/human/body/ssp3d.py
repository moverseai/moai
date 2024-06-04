import json
import logging
import os
import typing

import numpy as np
import torch

from moai.data.datasets.common.image2d import load_binary_image, load_color_image
from moai.utils.arguments import assert_path

log = logging.getLogger(__name__)

__all__ = ["SSP3D"]

# NOTE: Human3.6M body structure: {
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

# NOTE: COCO17 body structure: {
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
    def __init__(
        self,
        root: str,
        betas: int = 10,
        with_openpose: bool = False,
    ) -> None:
        super().__init__()
        assert_path(log, "EHF root path", root)
        labels = np.load(os.path.join(root, "labels.npz"))
        self.root = root
        self.betas = betas
        self.with_openpose = with_openpose
        for k, v in labels.items():
            setattr(
                self, k, torch.from_numpy(v) if np.issubdtype(v.dtype, np.number) else v
            )

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        data = {
            "color": load_color_image(
                os.path.join(self.root, "images", self.fnames[index])
            ),
            "silhouette": load_binary_image(
                os.path.join(self.root, "silhouettes", self.fnames[index])
            ),
            "shape": (
                torch.cat([self.shapes[index], torch.zeros(self.betas - 10)], dim=-1)
                if self.betas > 10
                else self.shapes[index]
            ),
            "global_orientation": self.poses[index][:3],
            "pose": self.poses[index][3:],
            "keypoints": self.joints2D[index][:, :2],
            "confidence": self.joints2D[index][:, 2:3],
            "camera_translation": self.cam_trans[index],
        }
        if self.with_openpose:
            filename = os.path.basename(self.fnames[index])
            kpts_all = self._read_keypoints(
                os.path.join(
                    self.root, "keypoints", filename.replace(".png", "_keypoints.json")
                )
            )["keypoints"]
            coco17 = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            coco25 = np.array([0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14])
            selected = min(
                kpts_all,
                key=lambda kpts: np.linalg.norm(
                    kpts[coco25][:, :2] - data["keypoints"][coco17]
                ).mean(),
            )
            data["openpose"] = {
                "keypoints": selected[:, :2],
                "confidence": selected[:, 2:3],
            }
        return data

    # TODO: refactor to use the same func (i.e. from openpose.py)
    def _read_keypoints(
        self,
        filename: str,
    ) -> typing.Dict[str, torch.Tensor]:
        with open(filename) as keypoint_file:
            data = json.load(keypoint_file)
        keypoints, gender_pd, gender_gt = [], [], []
        for person in data["people"]:
            body = np.array(person["pose_keypoints_2d"], dtype=np.float32)
            body = body.reshape([-1, 3])
            left_hand = np.array(
                person["hand_left_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])
            right_hand = np.array(
                person["hand_right_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])
            body = np.concatenate([body, left_hand, right_hand], axis=0)
            face = np.array(person["face_keypoints_2d"], dtype=np.float32).reshape(
                [-1, 3]
            )[17 : 17 + 51, :]
            contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
            body = np.concatenate([body, face, contour_keyps], axis=0)
            gender_pd.append(person.get("gender_pd", None))
            gender_gt.append(person.get("gender_gt", None))
            keypoints.append(torch.from_numpy(body))
        return {
            "keypoints": keypoints,
            "gender_pd": gender_pd,
            "gender_gt": gender_gt,
        }
