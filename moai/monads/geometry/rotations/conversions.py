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
    'RomaRotationVector2Quaternion',
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
            self.convert_func = kn.geometry.conversions.rotation_matrix_to_axis_angle
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
    __ALIAS_MAP__ = {
        'roma': dict(toolz.concat([
            ((R, 'rotmat') for R in ['R', 'rot', '3x3', 'rotation', 'matrix', 'rotation_matrix', 'rotmat']), 
            ((q, 'unitquat') for q in ['quat', 'quaternion', 'q', '4']),
            ((r, 'rotvec') for r in ['r', 'aa', 'rotvec', 'axisangle', 'rodrigues', 'axis-angle', '3']),
        ])),
        'kornia': dict(toolz.concat([
            ((R, 'rotation_matrix') for R in ['R', 'rot', '3x3', 'rotation', 'matrix', 'rotation_matrix', 'rotmat']), 
            ((q, 'quaternion') for q in ['quat', 'quaternion', 'q', '4']),
            ((r, 'angle_axis') for r in ['r', 'aa', 'rotvec', 'axisangle', 'rodrigues', 'axis-angle', '3']),
        ])),
    }

    __SHAPE_MAP__ = {
        'rotmat': [3, 3],
        'rotvec': [3],
        'unitquat': [4],
        'rotation_matrix': [3, 3],
        'angle_axis': [3],
        'quaternion': [4]
    }

    def __init__(self,
        src:        str, # one of aliases
        tgt:        str, # one of aliases
        backend:    str='roma',
    ):
        super().__init__()
        module = roma if backend == 'roma' else kn.geometry.conversions
        self.convert = getattr(module, f"{ConvertRotation.__ALIAS_MAP__[backend][src]}_to_{ConvertRotation.__ALIAS_MAP__[backend][tgt]}")
        self.shape = ConvertRotation.__SHAPE_MAP__[ConvertRotation.__ALIAS_MAP__[backend][src]]

    def forward(self,
        rotation:   torch.Tensor, # [B, R, 3, 3] for 'rotmat', [B, R, 3] for 'rotvec', [B, R, 4] for 'unitquat'
    ) -> torch.Tensor:
        return self.convert(rotation.view(rotation.shape[0], -1, *self.shape)) if not list(rotation.shape[-len(self.shape):]) == self.shape \
            else self.convert(rotation.contiguous())

# default is roma / kornia as fallback when roma does not deliver proper outputs
RotationMatrix2RotationVector = functools.partial(ConvertRotation, src='R', tgt='r', backend='kornia') # roma implementation is not deterministic / produces wrong outputs sometimes
RotationMatrix2Quaternion = functools.partial(ConvertRotation, src='R', tgt='q', backend='roma')

RotationVector2Quaternion = functools.partial(ConvertRotation, src='r', tgt='q', backend='roma')
RotationVector2RotationMatrix = functools.partial(ConvertRotation, src='r', tgt='R', backend='roma')

Quaternion2RotationVector = functools.partial(ConvertRotation, src='q', tgt='r', backend='roma')
Quaternion2RotationMatrix = functools.partial(ConvertRotation, src='q', tgt='R', backend='roma')

# roma
RomaRotationMatrix2RotationVector = functools.partial(ConvertRotation, src='R', tgt='r', backend='roma')
RomaRotationMatrix2Quaternion = functools.partial(ConvertRotation, src='R', tgt='q', backend='roma')

RomaRotationVector2Quaternion = functools.partial(ConvertRotation, src='r', tgt='q', backend='roma')
RomaRotationVector2RotationMatrix = functools.partial(ConvertRotation, src='r', tgt='R', backend='roma')

RomaQuaternion2RotationVector = functools.partial(ConvertRotation, src='q', tgt='r', backend='roma')
RomaQuaternion2RotationMatrix = functools.partial(ConvertRotation, src='q', tgt='R', backend='roma')

# kornia
KorniaRotationMatrix2RotationVector = functools.partial(ConvertRotation, src='R', tgt='r', backend='kornia')
KorniaRotationMatrix2Quaternion = functools.partial(ConvertRotation, src='R', tgt='q', backend='kornia')

KorniaRotationVector2Quaternion = functools.partial(ConvertRotation, src='r', tgt='q', backend='kornia')
KorniaRotationVector2RotationMatrix = functools.partial(ConvertRotation, src='r', tgt='R', backend='kornia')

KorniaQuaternion2RotationVector = functools.partial(ConvertRotation, src='q', tgt='r', backend='kornia')
KorniaQuaternion2RotationMatrix = functools.partial(ConvertRotation, src='q', tgt='R', backend='kornia')