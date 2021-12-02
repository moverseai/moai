import cv2
import torch
import torchvision
import typing
import numpy as np

def load_color_image(
    filename:       str,    
    output_space:   str='norm', # ['norm', 'ndc', 'pixel']
) -> torch.Tensor:
    img = torchvision.io.read_image(filename)
    if output_space == 'norm':
        img = img / 255.0
    return img

def load_depth_image(
    filename:   str,
    scale:      float=1.0,
) -> torch.Tensor:
    depth = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH)
    return torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) * scale

def load_normal_image(
    filename:   str,
    rotation:   typing.List[float]=[],
) -> torch.Tensor:
    pass

def load_mask_image(
    filename:   str,
) -> torch.Tensor:
    pass
