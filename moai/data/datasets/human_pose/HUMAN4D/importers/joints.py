import numpy
import torch

def load_skel3D_bin(
    filename: str,
    data_type = torch.float32,
    scale: float=1.0
):
    skel = numpy.load(filename).astype(numpy.float32) * scale
    num_people, num_joints, coords = skel.shape    
    return torch.from_numpy(
        skel.reshape(1, num_people, num_joints, coords)
    ).type(data_type)