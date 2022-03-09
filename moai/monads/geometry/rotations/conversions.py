import torch
import logging
import kornia as kn

log = logging.getLogger(__name__)

__all__ = [
    "Convert",
    "AxisAngle",
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