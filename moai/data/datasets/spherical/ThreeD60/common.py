import csv
import itertools
import cv2
import numpy
import torch
import torchvision
import enum
import logging
import typing

__all__ = [
    "Placements",
    "ImageTypes",
    "_load_paths",
    "_rotate_image",
    "_filename_separator",
    "_rotate_normal_image",
    "extract_image",
    "extract_path",
]

class Placements(enum.Enum):
    CENTER = 1    
    RIGHT = 2
    UP = 3

    def describe(self):        
        return self.name.lower(), self.value

    def __str__(self) -> str:
        n = str.lower(self.name) if self.value != 1 else str("left_down")
        return n

class ImageTypes(enum.Enum):
    COLOR = 1    
    DEPTH = 2
    NORMAL = 3

    def describe(self):        
        return self.name.lower(), self.value

    def __str__(self) -> str:
        return str.lower(self.name)        

_dataset_images_order = [
    str.format("{}_{}", str(Placements.CENTER), str(ImageTypes.COLOR)),
    str.format("{}_{}", str(Placements.RIGHT), str(ImageTypes.COLOR)),
    str.format("{}_{}", str(Placements.UP), str(ImageTypes.COLOR)),
    str.format("{}_{}", str(Placements.CENTER), str(ImageTypes.DEPTH)),
    str.format("{}_{}", str(Placements.RIGHT), str(ImageTypes.DEPTH)),
    str.format("{}_{}", str(Placements.UP), str(ImageTypes.DEPTH)),
    str.format("{}_{}", str(Placements.CENTER), str(ImageTypes.NORMAL)),
    str.format("{}_{}", str(Placements.RIGHT), str(ImageTypes.NORMAL)),
    str.format("{}_{}", str(Placements.UP), str(ImageTypes.NORMAL)),
]

_filename_separator = "_path"

def _create_selectors(
    placements: typing.Sequence[Placements],
    types: typing.Sequence[ImageTypes],
) -> typing.List[str]:
    return list(map( \
        lambda o: any(str(p) in o for p in placements) and any(str(t) in o for t in types), \
            _dataset_images_order))

def _load_paths(
    filename: str,
    name: str,
    placements: typing.Sequence[Placements],
    image_types: typing.Sequence[ImageTypes],
) -> typing.Dict[str, typing.Dict[str, str]]:
    selectors = _create_selectors(placements, image_types)
    with open (filename, 'r') as f:
        rows = [list(itertools.compress(row, selectors)) for row in csv.reader(f, delimiter=' ')]
        return list(map(
            lambda row: dict(list(
                (
                    str(p), dict(list(
                    (                        
                        str(t) + _filename_separator, \
                            next(filter(lambda r: str(p) in r.lower() and str(t) in r.lower(), row))
                    ) for t in image_types
                    ))
                ) for p in placements
            ))
        , [row for row in rows if all(map(lambda r: name in r, row))]
        ))

def _load_color_image( #TODO: extract these to a common module
    filename: str,
    data_type: torch.dtype
)  -> torch.Tensor:
    return torchvision.io.read_image(filename) / 255.0

def _load_depth_image( #TODO: extract these to a common module
    filename: str,
    data_type: torch.dtype
)  -> torch.Tensor:
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    h, w = depth_img.shape
    return torch.from_numpy(depth_img).type(data_type).reshape(1, h, w)

def _load_normal_image( #TODO: extract these to a common module
    filename: str,
    data_type: torch.dtype
)  -> torch.Tensor:#TODO: flip y-z channels
    normal_img = numpy.array(cv2.imread(filename, cv2.IMREAD_UNCHANGED)).transpose(2, 0, 1)
    c, h, w = normal_img.shape
    return torch.from_numpy(normal_img).reshape(c, h, w).type(data_type)

def _rotate_image(image: torch.Tensor, idx: int) -> torch.Tensor:
    width = image.shape[2]
    if idx < 0 or idx >= width:
        return
    rotated = torch.roll(image, idx, 2)
    return rotated

def _get_rotation_matrix(idx: int, width: int, dim: int=2) -> torch.Tensor:
    theta = (float(idx) / float(width)) * 2.0 * numpy.pi
    mat = [
        [1.0,       0.0,              0.0],\
        [0.0,       numpy.cos(theta),    numpy.sin(theta)],\
        [0.0,       numpy.sin(theta),    numpy.cos(theta)]]
    mat = numpy.asarray(mat, dtype = numpy.float32)
    mat = mat.transpose()
    return torch.as_tensor(torch.from_numpy(mat), dtype = torch.float32)

def _rotate_normal_image(image: torch.Tensor, idx: int) -> torch.Tensor:
    width = image.shape[2]
    if idx < 0 or idx >= width:
        return
    rot_mat = _get_rotation_matrix(idx, width)
    tmp_img = image.permute((1, 2, 0))
    rot_img = torch.matmul(tmp_img, rot_mat)
    rot_img = rot_img.permute((2, 0, 1))
    rot_img = torch.roll(rot_img, idx, 2)
    return rot_img
    
_image_loaders = {
    "color": lambda *params: _load_color_image(params[0], params[1]),
    "depth": lambda *params: _load_depth_image(params[0], params[1]),
    "normal": lambda *params: _load_normal_image(params[0], params[1]),
}

def _load_image(
    filename: str,
    image_type: ImageTypes,
    data_type: torch.dtype=torch.float32
) -> torch.Tensor:
    return _image_loaders[image_type](filename, data_type)

def extract_image(
    tensor_dict: typing.Dict[str, torch.Tensor],
    placement: Placements,
    image_type: ImageTypes
) -> torch.Tensor:
    if not str(placement) in tensor_dict or not str(image_type) in tensor_dict[str(placement)]:
        logging.fatal("Could not extract the requested placement/image (%s/%s) from the given tensor dictionary." % (placement, image_type))
    return tensor_dict[str(placement)][str(image_type)]

def extract_path(
    tensor_dict: typing.Dict[str, torch.Tensor],
    placement: Placements,
    image_type: ImageTypes
) -> str:
    if not str(placement) in tensor_dict or not str(image_type) + _filename_separator in tensor_dict[str(placement)]:
        logging.fatal("Could not extract the requested placement/image (%s/%s) path from the given tensor dictionary." % (placement, image_type))
    return tensor_dict[str(placement)][str(image_type) + _filename_separator]