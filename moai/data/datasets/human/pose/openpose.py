from moai.data.datasets.common import load_color_image

import torch
import glob
import typing
import logging
import json
import numpy as np
from collections import namedtuple

__all__ = [
    "OpenPoseInference",
]

logger = logging.getLogger(__name__)

OpenPoseParams = namedtuple('OpenPoseParams', ['load_hands', 'load_face', 'load_face_contour', 'single_person_only'])

class OpenPoseInference(torch.utils.data.Dataset):
    def __init__(self,
        image_glob:                 str,
        keypoints_glob:             str,
        load_hands:                 bool=False,
        load_face:                  bool=False,
        load_face_contour:          bool=False,
        single_person_only:         bool=False,
        invalid_joints:             typing.Sequence[int]=None,
    ):
        super(OpenPoseInference, self).__init__()
        self.params = OpenPoseParams(load_hands, load_face, load_face_contour, single_person_only)
        image_filenames = glob.glob(image_glob)
        keypoint_filenames = glob.glob(keypoints_glob)
        if len(image_filenames) != len(keypoint_filenames):
            logger.warning(
                f"Images ({len(image_filenames)}) and keypoints ({len(keypoint_filenames)}) counts differ. "
                f"Will continue with the smaller joint subset"
            )
        self.filenames = (image_filenames, keypoint_filenames)
        self.invalid_joints = list(invalid_joints)

    def __len__(self) -> int:
        return min(len(self.filenames[0]), len(self.filenames[0]))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        img_filename = self.filenames[0][index]
        keypoint_filename = self.filenames[1][index]
        img = load_color_image(img_filename)
        data = self._read_keypoints(keypoint_filename, 
            self.params.load_hands, self.params.load_face, self.params.load_face_contour
        )
        keypoints = data['keypoints']
        if self.params.single_person_only:
            def _get_area(keypoints: torch.Tensor) -> float:
                min_x = keypoints[..., 0].min()
                min_y = keypoints[..., 1].min()
                max_x = keypoints[..., 0].max()
                max_y = keypoints[..., 1].max()
                return (max_x - min_x) * (max_y - min_y) * keypoints[..., 2].sum()
            keypoints = [max(keypoints, key=_get_area)]
        keypoints = torch.stack(keypoints, dim=0).squeeze()
        # ones = torch.ones_like(keypoints[..., 0])
        keypoints[..., 2].scatter_(dim=0, 
            index=torch.Tensor(self.invalid_joints).long(), value=0.0
        )[:, np.newaxis]
        return {
            'color': img,
            'keypoints': keypoints[..., :2],
            'confidence': keypoints[..., 2][:, np.newaxis],
            'mask': (keypoints[..., 2] > 0.0).float()[:, np.newaxis]
        }

    def _read_keypoints(self,
        filename: str,
        load_hands=True,
        load_face=True,
        load_face_contour=False
    ) -> typing.Dict[str, torch.Tensor]:
        with open(filename) as keypoint_file:
            data = json.load(keypoint_file)
        keypoints, gender_pd, gender_gt = [], [], []
        for person in data['people']:
            body = np.array(person['pose_keypoints_2d'], dtype=np.float32)
            body = body.reshape([-1, 3])
            if load_hands:
                left_hand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
                right_hand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
                body = np.concatenate([body, left_hand, right_hand], axis=0)
            if load_face:
                face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
                contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
                if load_face_contour:
                    contour_keyps = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[:17, :]
                body = np.concatenate([body, face, contour_keyps], axis=0)

            gender_pd.append(person.get('gender_pd', None))
            gender_gt.append(person.get('gender_gt', None))
            keypoints.append(torch.from_numpy(body))
        return {
            'keypoints': keypoints,
            'gender_pd': gender_pd,
            'gender_gt': gender_gt, 
        }