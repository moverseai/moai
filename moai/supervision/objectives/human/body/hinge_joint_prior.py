import numpy as np
import torch

#NOTE: taken from vchoutas/smplify-x

__all__ = [
    'HingeJointPrior',
]

class HingeJointPrior(torch.nn.Module):
    def __init__(self,
        with_global_pose:       bool=True,
    ):
        super(HingeJointPrior, self).__init__()
        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2        
        self.register_buffer('angle_prior_idxs', torch.from_numpy(
            np.array([55, 58, 12, 15], dtype=np.int64)
        ))
        self.register_buffer('angle_prior_signs', torch.from_numpy(
            np.array([1, -1, -1, -1], dtype=np.float32)
        ))
        self.with_global_pose = with_global_pose

    def forward(self, 
        pose:       torch.Tensor
    ) -> torch.Tensor:
        ''' Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not self.with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] * self.angle_prior_signs).pow(2)