from moai.data.datasets.common import load_color_image

import os
import torch
import glob
import typing
import logging
import json
import numpy as np
from collections import namedtuple

from moai.utils.arguments.path import assert_path

__all__ = [
    "OpenPoseInference",
    "OpenPoseKeypoints",
    "OpenPoseMultiviewKeypoints",
]

log = logging.getLogger(__name__)

OpenPoseParams = namedtuple('OpenPoseParams', [
    'load_hands',
    'load_face',
    'load_face_contour',
    'single_person_only',
])

class OpenPoseInferred(torch.utils.data.Dataset):
    def __init__(self,
        image_glob:                 str,
        load_hands:                 bool=False,
        load_face:                  bool=False,
        load_face_contour:          bool=False,
        single_person_only:         bool=False,
        invalid_joints:             typing.Sequence[int]=None,
        # updates:                    typing.Mapping[str, bool]={},
    ):
        super(OpenPoseInferred, self).__init__()
        self.params = OpenPoseParams(load_hands, 
            load_face, load_face_contour, single_person_only,
        )
        image_filenames = glob.glob(image_glob)
        self.filenames = image_filenames
        self.invalid_joints = list(invalid_joints or [])
        log.info(f"Loaded {len(self.filenames)} OpenPose inference results.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        img_filename = self.filenames[index]
        folder, filename = os.path.split(img_filename)
        base_dir = os.path.basename(folder)
        base_dir = 'keypoints' if base_dir == 'images' else base_dir
        name, ext = os.path.splitext(filename)
        img = load_color_image(img_filename)
        keypoint_filename = os.path.join(os.path.dirname(folder), base_dir, f"{name}_keypoints.json")
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
        keypoints[..., 2].scatter_(dim=0, 
            index=torch.Tensor(self.invalid_joints).long(), value=0.0
        )[:, np.newaxis]
        kpts = keypoints[..., :2]
        conf = keypoints[..., 2][:, np.newaxis]
        return {
            'color': img,
            'keypoints': kpts,
            'confidence': conf,
            'mask': (conf > 0.0).float()[:, np.newaxis]
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

            gender_pd.append(person.get('gender_pd', None) or person.get('gender', None))
            gender_gt.append(person.get('gender_gt', None))
            keypoints.append(torch.from_numpy(body))
        return {
            'keypoints': keypoints,
            'gender_pd': gender_pd,
            'gender_gt': gender_gt, 
        }

class OpenPoseKeypoints(torch.utils.data.Dataset):

    __ALL_TYPES__ = ['body', 'face', 'hands', 'face_contour', 'gender']

    def __init__(self,
        root:           str,
        types:          typing.Union[typing.List[str], str]='all',
        single_person:  bool=True,
    ) -> None:
        assert_path(log, 'openpose root', root)        
        self.filenames = glob.glob(os.path.join(root, '*.json'))
        self.types = OpenPoseKeypoints.__ALL_TYPES__ if types == 'all' or types == '**' else types
        self.single_person = single_person
        log.info(f"Loaded {len(self.filenames)} OpenPose keypoint estimations.")

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_keypoints(self, filename: str) -> dict:
        with open(filename) as keypoint_file:
            data = json.load(keypoint_file)
        return data

    def _extract_data(self, data: dict) -> typing.Dict[str, torch.Tensor]:
        persons = []
        for person in data['people']:
            persons.append({})
            body = np.array(person['pose_keypoints_2d'], dtype=np.float32)
            body = torch.from_numpy(body.reshape([-1, 3]))
            persons[-1]['body'] = { 'keypoints': body[:, :2], 'confidence': body[:, 2:3] }
            persons[-1]['all'] = { 'keypoints': body[:, :2], 'confidence': body[:, 2:3] }
            if 'hands' in self.types:
                left_hand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
                right_hand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
                # body = np.concatenate([body, left_hand, right_hand], axis=0)
                left_hand = torch.from_numpy(left_hand)
                right_hand = torch.from_numpy(right_hand)
                persons[-1]['hands'] = { 
                    'left': {
                        'keypoints': left_hand[:, :2],
                        'confidence': left_hand[:, 2:3],
                     },
                     'right': {
                        'keypoints': right_hand[:, :2],
                        'confidence': right_hand[:, 2:3],
                     }  
                }
                persons[-1]['all']['keypoints'] = torch.cat([
                    persons[-1]['all']['keypoints'], 
                    persons[-1]['hands']['left']['keypoints'],
                    persons[-1]['hands']['right']['keypoints'],
                ], dim=0)
                persons[-1]['all']['confidence'] = torch.cat([
                    persons[-1]['all']['confidence'], 
                    persons[-1]['hands']['left']['confidence'],
                    persons[-1]['hands']['right']['confidence'],
                ], dim=0)
            if 'face' in self.types:
                face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
                face = torch.from_numpy(face)
                persons[-1]['face'] = { 'keypoints': face[:, :2], 'confidence': face[:, 2:3] }
                # contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
                persons[-1]['all']['keypoints'] = torch.cat([
                    persons[-1]['all']['keypoints'], 
                    persons[-1]['face']['keypoints'],
                ], dim=0)
                persons[-1]['all']['confidence'] = torch.cat([
                    persons[-1]['all']['confidence'], 
                    persons[-1]['face']['confidence'],
                ], dim=0)
                if 'face_contour' in self.types:
                    contour_keyps = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[:17, :]
                    contour_keyps = torch.from_numpy(contour_keyps)
                    persons[-1]['face_contour'] = { 'keypoints': contour_keyps[:, :2], 'confidence': contour_keyps[:, 2:3] }
                    persons[-1]['all']['keypoints'] = torch.cat([
                        persons[-1]['all']['keypoints'], 
                        persons[-1]['face_contour']['keypoints'],
                    ], dim=0)
                    persons[-1]['all']['confidence'] = torch.cat([
                        persons[-1]['all']['confidence'], 
                        persons[-1]['face_contour']['confidence'],
                    ], dim=0)
                # body = np.concatenate([body, face, contour_keyps], axis=0)
            if 'gender' in self.types:
                gender = person.get('gender_gt', None) or person.get('gender', None) or person.get('gender_pd', None)
                if gender is not None:
                    persons[-1]['gender']
        if not self.single_person:
            return persons
        else:            
            def _get_area(person: typing.Dict[str, torch.Tensor]) -> float:
                keypoints = person['body']['keypoints']
                confidence = person['body']['confidence']
                #TODO: update selection with threshold based discarding?
                min_x = keypoints[..., 0].min()
                min_y = keypoints[..., 1].min()
                max_x = keypoints[..., 0].max()
                max_y = keypoints[..., 1].max()
                return (max_x - min_x) * (max_y - min_y) * confidence.sum()
            person = max(persons, key=_get_area)
            return person

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        filename = self.filenames[index]
        data = self._load_keypoints(filename)
        return self._extract_data(data)

class OpenPoseMultiviewKeypoints(OpenPoseKeypoints):
    def __init__(self,
        root:           str,
        views:          typing.Union[typing.List[str], str],
        types:          typing.Union[typing.List[str], str]='all',
        single_person:  bool=True,
    ):
        assert_path(log, 'openpose root', root)
        if isinstance(views, str) and (views == 'all' or views == '**'):
            views = [d for d in os.listdir(root) if os.path.isdir(d)]
        self.filenames = { }
        for view in views:
            self.filenames[view] = glob.glob(os.path.join(root, view, "*.json"))
        self.single_person = single_person
        self.types = OpenPoseMultiviewKeypoints.__ALL_TYPES__ if types == 'all' or types == '**' else types
        
    def __len__(self) -> int:
        return len(next(iter(self.filenames.values()), []))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        data = { 'poses2d': { }}
        for k, v in self.filenames.items():
            data['poses2d'][k] = super()._extract_data(super()._load_keypoints(v[index]))
        return data