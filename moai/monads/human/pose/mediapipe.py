import torch
import functools
import numpy as np
import logging

__all__ = [
    "JointMap",
]

log = logging.getLogger(__name__)

def _hand_to_mediapipe(
    model_type:         str='mano',
    format:             str='mediapipe_hand',
    with_tips:          bool=False, 
) -> np.array:
    ''' Returns the indices of the permutation that maps MediaPipe to MANO

        Parameters
        ----------
        model_type: str, optional
            The type of MANO-like model that is used. The default mapping
            returned is for the SMPLX model
        with_tips: bool, optional
            Flag for adding to the returned permutation the mapping for the
            finger tips keypoints. Defaults to False

    '''
    if format.lower() == 'mediapipe_hand':
        if model_type == 'mano':
            if with_tips:
                return np.array([
                    0, 13, 14, 15, 16,
                    1, 2, 3, 17,
                    4, 5, 6, 18,
                    10, 11, 12, 19,
                    7, 8, 9, 20
                ], dtype=np.int32)
            else:
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
        with_tips:          bool=False,
    ):
        super(JointMap, self).__init__()
        self.register_buffer('indices', torch.from_numpy(
            _hand_to_mediapipe(model_type=model, 
                format=format, 
                with_tips=with_tips,
            )).long()
        )
        self.index = functools.partial(torch.index_select, dim=1)

    def forward(self, 
        joints: torch.Tensor,
    ) -> torch.Tensor:
        return self.index(joints, index=self.indices)