import numpy
import torch

__all__ = ['load_extrinsics']

def load_extrinsics(
    filename: str,
    data_type=torch.float32,
    translation_scale: float=1e-03,
):
    data = numpy.loadtxt(filename)
    pose = numpy.zeros([4, 4])
    pose[3, 3] = 1
    pose[:3, :3] = data[:3, :3]
    pose[:3, 3] = data[3, :] * translation_scale
    #y = R*x + t
    rotation = torch.tensor(data[:3, :3], dtype=data_type)
    translation = torch.tensor(data[3, :], dtype=data_type) * translation_scale
    #y = Rx + t <-> x = Rinv*y - Rinv * t 
    rotation_inverse = torch.tensor(numpy.linalg.inv(pose[:3, :3]), dtype=data_type)
    translation_inverse = torch.tensor(numpy.inner(rotation_inverse, pose[:3, 3]) * (-1), dtype=data_type)
    #Continue after walk
    return rotation, translation, rotation_inverse, translation_inverse