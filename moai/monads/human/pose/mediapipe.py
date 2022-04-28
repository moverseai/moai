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

# __JOINT_FORMATS__ = {
#     'none':                 [118,   0,          0],
#     'coco25':               [25,    21 * 2,     51],
#     'coco25_star':          [25,    0,          0],
#     'coco25_star+':         [25,    21 * 2,     0],
# }

# class Split(torch.nn.Module):
#     def __init__(self,
#         format:             str='coco25',
#     ):
#         super(Split, self).__init__()
#         self.split_sections = __JOINT_FORMATS__[format]

#     def forward(self,
#         all_joints:          torch.Tensor=None,
#     ) -> typing.Mapping[str, torch.Tensor]:
#         s = sum(self.split_sections)
#         body, hands, face = torch.split(all_joints[..., :s, :], self.split_sections, dim=-2)
#         return {
#             'body':     body,
#             'hands':    hands,
#             'face':     face,
#         }

def _hand_to_mediapipe(
    model_type:         str='mano',
    format:             str='mediapipe_hand',
    use_tips:           bool=False,
    use_face:           bool=True,
    use_face_contour:   bool=False,    
) -> np.array:
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
    # if openpose_format.lower() == 'coco25':
    if format.lower() == 'mediapipe_hand':
        if model_type == 'mano':
            return np.array([
                0, 13, 14, 15, 15,
                1, 2, 3, 3,
                4, 5, 6, 6,
                10, 11, 12, 12,
                7, 8, 9, 9
            ], dtype=np.int32)
    else:
        log.error(f'Unknown hand format: {format}')

class JointMap(torch.nn.Module):
    def __init__(self, 
        model:              str='mano',
        format:             str='mediapipe_hand',
        with_tips:         bool=False,
        with_face:          bool=False,
        with_face_contour:  bool=False,
    ):
        super(JointMap, self).__init__()
        self.register_buffer('indices', torch.from_numpy(
            _hand_to_mediapipe(model_type=model, 
                format=format, 
                use_tips=with_tips,
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

# class JointConfidence(torch.nn.Module):
#     def __init__(self,
#         joints_to_ignore:       typing.Sequence[int],
#         confidence_threshold:   float=0.0,
#     ):
#         super(JointConfidence, self).__init__()
#         self.register_buffer('ignore', torch.tensor(joints_to_ignore).long())
#         self.threshold = confidence_threshold

#     def forward(self, 
#         confidence: torch.Tensor
#     ) -> torch.Tensor:
#         ret = confidence.clone()
#         ret[:, self.ignore, ...] = 0.0
#         ret[ret < self.threshold] = 0.0
#         return ret

# class MergeToes(torch.nn.Module): #TODO: aug layer
#     def __init__(self):
#         super(MergeToes, self).__init__()

#     def forward(self,
#         keypoints:  torch.Tensor,
#         confidence: torch.Tensor,
#     ) -> typing.Dict[str, torch.Tensor]:
#         kpts = keypoints.detach().clone()
#         conf = confidence.detach().clone()
#         # right
#         right_toes_j = kpts[:, 22:24, :]
#         right_toes_w = conf[:, 22:24, :]
#         right_toe_w = right_toes_w.sum(dim=1, keepdim=True)
#         right_toe = (right_toes_j * right_toes_w).sum(dim=1, keepdim=True) / (right_toe_w + 1e-8)
#         kpts[:, 22, :] = right_toe
#         kpts[:, 23, :] = right_toe
#         right_toe_w = right_toe_w * 0.5
#         conf[:, 22, :] = right_toe_w
#         conf[:, 23, :] = right_toe_w
#         # left
#         left_toes_j = kpts[:, 19:21, :]
#         left_toes_w = conf[:, 19:21, :]
#         left_toe_w = left_toes_w.sum(dim=1, keepdim=True)
#         left_toe = (left_toes_j * left_toes_w).sum(dim=1, keepdim=True) / (left_toe_w + 1e-8)
#         kpts[:, 19, :] = left_toe
#         kpts[:, 20, :] = left_toe
#         left_toe_w = left_toe_w * 0.5
#         conf[:, 19, :] = left_toe_w
#         conf[:, 20, :] = left_toe_w
#         return {
#             'positions' : kpts,
#             'confidence': conf,
#         }
