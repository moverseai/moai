import json
import numpy
import torch

__all__ = [
    'load_intrinsics_repository',
    'get_intrinsics',
]

def load_intrinsics_repository(filename: str, stream: str):    
    '''
        use color for loading color intrinsics and depth for depth intrinsics
    '''
    with open(filename, 'r') as json_file:
        intrinsics_repository = json.load(json_file)
        if stream == "depth":
            intrinsics_dict = dict((intrinsics['Device'], \
                intrinsics['Depth Intrinsics'][0]['1280x720'])\
                    for intrinsics in intrinsics_repository)
        if stream == "color":
            intrinsics_dict = dict((intrinsics['Device'], \
                intrinsics['Depth Intrinsics'][0]['1280x720'])\
                    for intrinsics in intrinsics_repository)
    return intrinsics_dict

def get_intrinsics(name, intrinsics_dict, scale=1, center_crop={'width': 0, 'height': 0}, data_type=torch.float32):
    if intrinsics_dict is not None:
        intrinsics_data = numpy.array(intrinsics_dict[name])
        intrinsics = torch.tensor(intrinsics_data).reshape(3, 3).type(data_type)   
        intrinsics[0, 2] = intrinsics[0, 2] - center_crop['width']
        intrinsics[1, 2] = intrinsics[1, 2] - center_crop['height']

        intrinsics[0, 0] = intrinsics[0, 0] / scale
        intrinsics[0, 2] = intrinsics[0, 2] / scale
        intrinsics[1, 1] = intrinsics[1, 1] / scale
        intrinsics[1, 2] = intrinsics[1, 2] / scale
        intrinsics_inv = intrinsics.inverse()
        return intrinsics, intrinsics_inv
    raise ValueError("Intrinsics repository is empty")    