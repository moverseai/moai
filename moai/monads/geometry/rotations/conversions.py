import torch
import logging
import kornia as kn
import roma
import toolz
import functools

log = logging.getLogger(__name__)

__all__ = [
    "Convert",
    "AxisAngle",
    "RotMat",
    'ConvertRotation',
    'RotationMatrix2Quaternion',
    'RotationMatrix2RotationVector',
    'RotationVector2Quaternion',
    'RotationVector2RotationMatrix',
    'Quaternion2RotationMatrix',
    'Quaternion2RotationVector',
]

#TODO: make_from, convert_to
class AxisAngle(torch.nn.Module):
    def __init__(self,
        to:    str = "mat", #"quat" , "ortho6d", "euler_angles" , etc.        
    ):
        super().__init__()
        if to == "mat":
            self.convert_func = kn.geometry.conversions.angle_axis_to_rotation_matrix
        elif to == "quat":
            #(x, y, z, w) format of quaternion
            self.convert_func = kn.geometry.conversions.angle_axis_to_quaternion
        else:
            log.error(f"The selected rotational convertion {to} is not supported. Please enter a valid one to continue.")

    def forward(self,angle_axis: torch.Tensor) -> torch.Tensor:
        return self.convert_func(angle_axis)

class RotMat(torch.nn.Module):
    def __init__(self,
        to:    str = "axis_angle", #"quat" , "ortho6d", "euler_angles" , etc.        
    ):
        super().__init__()
        if to == "axis_angle":
            self.convert_func = kn.geometry.conversions.rotation_matrix_to_angle_axis
        elif to == "quat":
            #(x, y, z, w) format of quaternion
            self.convert_func = kn.geometry.conversions.rotation_matrix_to_quaternion
        else:
            log.error(f"The selected rotational convertion {to} is not supported. Please enter a valid one to continue.")

    def forward(self,rot_mat: torch.Tensor) -> torch.Tensor:
        return self.convert_func(rot_mat)

class Convert(torch.nn.Module):
    __MAPPING__ = {
        'aa': AxisAngle,        
    }

    def __init__(self,
        from_to:       str,
    ):
        super().__init__()
        f, t = from_to.split('_')
        self.converter = Convert.__MAPPING__[f](to=t)

    def forward(self, rotation: torch.Tensor) -> torch.Tensor:
        return self.converter(rotation)

class ConvertRotation(torch.nn.Module):
    __ALIAS_MAP__ = dict(toolz.concat([
        ((R, 'rotmat') for R in ['R', 'rot', '3x3', 'rotation', 'matrix', 'rotation_matrix', 'rotmat']), 
        ((q, 'unitquat') for q in ['quat', 'quaternion', 'q', '4'])
        ((r, 'rotvec') for r in ['r', 'aa', 'rotvec', 'axisangle', 'rodrigues', 'axis-angle', '3'])
    ]))

    def __init__(self,
        src:        str, # one of aliases
        tgt:        str, # one of aliases
    ):
        super().__init__()
        self.convert = getattr(roma, f"{ConvertRotation.__ALIAS_MAP__[src]}_to_{ConvertRotation.__ALIAS_MAP__[tgt]}")

    def forward(self,
        rotation:   torch.Tensor, # [B, R, 3, 3] for 'rotmat', [B, R, 3] for 'rotvec', [B, R, 4] for 'unitquat'
    ) -> torch.Tensor:
        return self.convert(rotation)

RotationMatrix2RotationVector = functools.partia(ConvertRotation, src='R', tgt='r')
RotationMatrix2Quaternion = functools.partia(ConvertRotation, src='R', tgt='q')

RotationVector2Quaternion = functools.partia(ConvertRotation, src='r', tgt='q')
RotationVector2RotationMatrix = functools.partia(ConvertRotation, src='r', tgt='R')

Quaternion2RotationVector = functools.partia(ConvertRotation, src='q', tgt='r')
Quaternion2RotationMatrix = functools.partia(ConvertRotation, src='q', tgt='R')