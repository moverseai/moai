import cv2
import torch
import numpy

__all__ = [
    'load_image',
    'load_depth',
    'load_normal',
]

def load_image(
    filename: str, 
    data_type: torch.TensorType=torch.float32,
) -> torch.Tensor:
    color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
    h, w, c = color_img.shape
    color_data = color_img.astype(numpy.float32).transpose(2, 0, 1)
    return torch.from_numpy(
        color_data.reshape(1, c, h, w)        
    ).type(data_type) / 255.0

def load_depth(
    filename: str,
    data_type: torch.TensorType=torch.float32, 
    scale: float=1.0,
) -> torch.Tensor:
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    h, w = depth_img.shape
    depth_data = depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

def load_normal(
    filename: str,
    data_type: torch.TensorType=torch.float32
) -> torch.Tensor:
    normal_img = numpy.array(cv2.imread(filename, cv2.IMREAD_UNCHANGED)).transpose(2, 0, 1)
    c, h, w = normal_img.shape
    return torch.from_numpy(normal_img).reshape(1,c, h, w).type(data_type)