import torch
import logging
import kornia as kn

log = logging.getLogger(__name__)

__all__ = ["AngleAxis"]

#TODO: make_from, convert_to
class AngleAxis(torch.nn.Module):
    def __init__(self,
        convert_to:    str = "rot_mat", #"quat" , "ortho6d", "euler_angles" , etc.        
    ):
        super(AngleAxis,self).__init__()
        if convert_to == "rot_mat":
            self.convert_func = kn.geometry.conversions.angle_axis_to_rotation_matrix
        elif convert_to == "quat":
            #(x, y, z, w) format of quaternion
            self.convert_func = kn.geometry.conversions.angle_axis_to_quaternion
        else:
            log.error(f"The selected rotational convertion {convert_to} is not supported. Please enter a valid one to continue.")

    def forward(self,angle_axis: torch.Tensor) -> torch.Tensor:
        return self.convert_func(angle_axis)