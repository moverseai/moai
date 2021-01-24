from moai.data.datasets.spherical.ThreeD60.matterport3d import Matterport3D
from moai.data.datasets.spherical.ThreeD60.stanford2d3d import Stanford2D3D
from moai.data.datasets.spherical.ThreeD60.suncg import SunCG
from moai.data.datasets.spherical.ThreeD60.rotation import RotationDataset
from moai.data.datasets.spherical.ThreeD60.common import (
    Placements,
    ImageTypes,
    extract_image,
    extract_path
)

import typing
import torch

__all__ = ["ThreeD60"]

_dataset_generators = {
    "suncg": lambda *params: SunCG(params[0], params[1], params[2]),
    "m3d": lambda *params: Matterport3D(params[0], params[1], params[2]),
    "s2d3d": lambda *params: Stanford2D3D(params[0], params[1], params[2])
}

def _get_datasets(filename, datasets=["suncg", "s2d3d", "m3d"],              \
        placements=[Placements.CENTER, Placements.RIGHT, Placements.UP],    \
        image_types=[ImageTypes.COLOR, ImageTypes.DEPTH, ImageTypes.NORMAL], 
        longitudinal_rotation = False):
    return torch.utils.data.ConcatDataset(list(map(                                                      \
        lambda d: RotationDataset(_dataset_generators[d](filename, placements, image_types)) 
                if longitudinal_rotation else _dataset_generators[d](filename, placements, image_types), datasets)))

class ThreeD60(torch.utils.data.Dataset):
    def __init__(self,
        filename: str,
        datasets: typing.Sequence[str]=["suncg", "s2d3d", "m3d"],
        placements: typing.Sequence[str]=[Placements.CENTER, Placements.RIGHT, Placements.UP],
        image_types: typing.Sequence[str]=[ImageTypes.COLOR, ImageTypes.DEPTH, ImageTypes.NORMAL],
        longitudinal_rotation: bool=False
    ):
        super(ThreeD60, self).__init__()
        enum_placements = list(map(lambda p: 
            Placements[p.upper()], placements)
        )
        enum_image_types = list(map(lambda t: 
            ImageTypes[t.upper()], image_types)
        )
        self.iterator = _get_datasets(filename, datasets, 
            enum_placements, enum_image_types, longitudinal_rotation
        )

    def __len__(self) -> int:
        return len(self.iterator)

    def __getitem__(self, index) -> typing.Dict[str, torch.Tensor]:
        return self.iterator[index]