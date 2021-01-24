from moai.data.datasets.spherical.ThreeD60.common import (
    _load_paths,
    _load_image,
    _filename_separator,
    Placements,
    ImageTypes,
)

import os
import copy
import logging
import torch
import typing

log = logging.getLogger(__name__)

__all__ = ["SunCG"]

class SunCG(torch.utils.data.Dataset):
    def __init__(self, 
        filename: str,
        placements: typing.Sequence[Placements],
        image_types: typing.Sequence[ImageTypes]
    ):
        assert(os.path.exists(filename))
        super(SunCG, self).__init__()
        self.entries = _load_paths(filename, type(self).__name__, placements, image_types)
        log.info(f"Loaded {len(self.entries)} items from SunCG.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        output = { }
        for placement, type_map in self.entries[index].items():
            for typed_path, filename in type_map.items():
                image_type = typed_path.replace(_filename_separator, "")
                output[placement + "_" + image_type] = _load_image(filename, image_type)
        return output