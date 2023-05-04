import torch
import typing
import functools
import numpy as np
import logging

#NOTE: code from https://github.com/vchoutas/smplify-x

__all__ = [
    "Split",
    "JointMap",
    "JointConfidence",
    "MergeToes",
]

log = logging.getLogger(__name__)

__JOINT_FORMATS__ = {
    'none':                 [118,   0,          0],
    'coco25':               [25,    21 * 2,     51],
    'coco25_face':          [25,    21 * 2,     51 + 17],
    'coco25_star':          [25,    0,          0],
    'coco25_star+':         [25,    21 * 2,     0],
}

class Split(torch.nn.Module):
    def __init__(self,
        format:             str='coco25',
    ):
        super(Split, self).__init__()
        self.split_sections = __JOINT_FORMATS__[format]

    def forward(self,
        joints:          torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        s = sum(self.split_sections)
        body, hands, face = torch.split(joints[..., :s, :], self.split_sections, dim=-2)
        return {
            'body':     body,
            'hands':    hands,
            'face':     face,
        }

def _body_to_openpose(
    model_type:         str='smplx',
    openpose_format:    str='coco25',
    use_hands:          bool=True,
    use_face:           bool=True,
    use_face_contour:   bool=False,    
) -> np.ndarray:
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([
                24, 12, 17, 19, 21,
                16, 18, 20, 0, 2,
                5, 8, 1, 4, 7,
                25, 26, 27, 28, 29,
                30, 31, 32, 33, 34
            ], dtype=np.int32)        
        elif model_type == 'smplh':
            body_mapping = np.array([
                52, 12, 17, 19, 21,
                16, 18, 20, 0, 2,
                5, 8, 1, 4, 7,
                53, 54, 55, 56, 57,
                58, 59, 60, 61, 62
            ], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([
                    20, 34, 35, 36, 63,
                    22, 23, 24, 64, 25,
                    26, 27, 65, 31, 32,
                    33, 66, 28, 29, 30,
                    67
                ], dtype=np.int32)
                rhand_mapping = np.array([
                    21, 49, 50, 51, 68,
                    37, 38, 39, 69, 40,
                    41, 42, 70, 46, 47, 
                    48, 71, 43, 44, 45,
                    72
                ], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([
                55, 12, 17, 19, 21,
                16, 18, 20, 0, 2,
                5, 8, 1, 4, 7, 
                56, 57, 58, 59, 60,
                61, 62, 63, 64, 65
            ], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([
                    20, 37, 38, 39, 66,
                    25, 26, 27, 67, 28,
                    29, 30, 68, 34, 35,
                    36, 69, 31, 32, 33, 
                    70
                ], dtype=np.int32)
                rhand_mapping = np.array([
                    21, 52, 53, 54, 71,
                    40, 41, 42, 72, 43,
                    44, 45, 73, 49, 50,
                    51, 74, 46, 47, 48,
                    75
                ], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour, dtype=np.int32)
                mapping += [face_mapping]
            return np.concatenate(mapping)
        elif model_type == 'star':
            body_mapping = np.array([
                15, 12, 17, 19, 21, 
                16, 18, 20, 0, 2, 
                5, 8, 1, 4, 7, 
                24, 25, 26, 27, 10,
                10, 7, 11, 11, 8,
            ], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 22,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0
                ], dtype=np.int32)
                rhand_mapping = np.array([
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 23,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0
                ], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face: 
                #NOTE: here we place the face landmarks to the original body mapping
                mapping[0][15] = 25 # REye
                mapping[0][16] = 26 # LEye
                mapping[0][17] = 24 # REar
                mapping[0][18] = 27 # LEar
                #NOTE: currently not used anywhere, just a placeholder for the face landmarks
                # face_mapping = np.arange(76, 127 + 17 * use_face_contour, dtype=np.int32)
                # mapping += [face_mapping]
            return np.concatenate(mapping)
        else:
            log.error(f'Unknown model type: {model_type}')
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([
                24, 12, 17, 19, 21,
                16, 18, 20, 0, 2,
                5, 8, 1, 4, 7,
                25, 26, 27, 28
            ], dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([
                52, 12, 17, 19, 21,
                16, 18, 20, 0, 2, 
                5, 8, 1, 4, 7,
                53, 54, 55, 56
            ], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([
                    20, 34, 35, 36, 57,
                    22, 23, 24, 58, 25,
                    26, 27, 59, 31, 32,
                    33, 60, 28, 29, 30,
                    61
                ], dtype=np.int32)
                rhand_mapping = np.array([
                    21, 49, 50, 51, 62,
                    37, 38, 39, 63, 40,
                    41, 42, 64, 46, 47,
                    48, 65, 43, 44, 45,
                    66
                ], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([
                55, 12, 17, 19, 21,
                16, 18, 20, 0, 2,
                5, 8, 1, 4, 7,
                56, 57, 58, 59
            ], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([
                    20, 37, 38, 39, 60,
                    25, 26, 27, 61, 28,
                    29, 30, 62, 34, 35,
                    36, 63, 31, 32, 33,
                    64
                ], dtype=np.int32)
                rhand_mapping = np.array([
                    21, 52, 53, 54, 65,
                    40, 41, 42, 66, 43,
                    44, 45, 67, 49, 50,
                    51, 68, 46, 47, 48,
                    69
                ], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 + 17 * use_face_contour, dtype=np.int32)
                mapping += [face_mapping]
            return np.concatenate(mapping)
        else:
            log.error(f'Unknown model type: {model_type}')
    else:
        log.error(f'Unknown joint format: {openpose_format}')

class JointMap(torch.nn.Module):
    def __init__(self, 
        model:              str='smplx',
        format:             str='coco25',
        with_hands:         bool=False,
        with_face:          bool=False,
        with_face_contour:  bool=False,
    ):
        super(JointMap, self).__init__()
        self.register_buffer('indices', torch.from_numpy(
            _body_to_openpose(model_type=model, 
                openpose_format=format, 
                use_hands=with_hands,
                use_face=with_face,
                use_face_contour=with_face_contour,
            )).long()
        )
        self.index = functools.partial(torch.index_select, dim=1)

    def forward(self, 
        joints: torch.Tensor,
        vertices: torch.Tensor=None, #NOTE: for compatibility with current SMPLX implementation
    ) -> torch.Tensor:
        return self.index(joints, index=self.indices)

class JointConfidence(torch.nn.Module):
    def __init__(self,
        joints_to_ignore:       typing.Sequence[int],
        confidence_threshold:   float=0.0,
    ):
        super(JointConfidence, self).__init__()
        self.register_buffer('ignore', torch.tensor(joints_to_ignore).long())
        self.threshold = confidence_threshold

    def forward(self, 
        confidence: torch.Tensor
    ) -> torch.Tensor:
        ret = confidence.clone()
        if len(ret.shape) == 3:
            ret[:, self.ignore, ...] = 0.0
        else:
           ret[self.ignore, ...] = 0.0 
        ret[ret < self.threshold] = 0.0
        return ret

class MergeToes(torch.nn.Module): #TODO: aug layer
    def __init__(self):
        super(MergeToes, self).__init__()

    def forward(self,
        keypoints:  torch.Tensor,
        confidence: torch.Tensor,
    ) -> typing.Dict[str, torch.Tensor]:
        kpts = keypoints.detach().clone()
        conf = confidence.detach().clone()
        # right
        right_toes_j = kpts[:, 22:24, :]
        right_toes_w = conf[:, 22:24, :]
        right_toe_w = right_toes_w.sum(dim=1, keepdim=True)
        right_toe = (right_toes_j * right_toes_w).sum(dim=1, keepdim=True) / (right_toe_w + 1e-8)
        kpts[:, 22:23, :] = right_toe
        kpts[:, 23:24, :] = right_toe
        right_toe_w = right_toe_w * 0.5
        conf[:, 22:23, :] = right_toe_w
        conf[:, 23:24, :] = right_toe_w
        # left
        left_toes_j = kpts[:, 19:21, :]
        left_toes_w = conf[:, 19:21, :]
        left_toe_w = left_toes_w.sum(dim=1, keepdim=True)
        left_toe = (left_toes_j * left_toes_w).sum(dim=1, keepdim=True) / (left_toe_w + 1e-8)
        kpts[:, 19:20, :] = left_toe
        kpts[:, 20:21, :] = left_toe
        left_toe_w = left_toe_w * 0.5
        conf[:, 19:20, :] = left_toe_w
        conf[:, 20:21, :] = left_toe_w
        return {
            'positions' : kpts,
            'confidence': conf,
        }
