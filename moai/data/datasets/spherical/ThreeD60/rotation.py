from moai.data.datasets.spherical.ThreeD60.common import (
    _rotate_image,
    _rotate_normal_image,
    ImageTypes
)

import torch
import random
import typing

__all__ = ["RotationDataset"]

#TODO: extract this to augmentation layer
class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, super_dataset):
        super(RotationDataset, self).__init__()
        self.super = super_dataset
        
    def __len__(self) -> int:
        return len(self.super)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        entries = self.super[index]
        width = entries['left_down']['color'].shape[2]
        idx = random.randint(0, width - 1)
        for placement, type_map in entries.items():
            if 'color' in type_map:
                entries[placement]['color'] = _rotate_image(entries[placement]['color'], idx)
            if 'depth' in type_map:
                entries[placement]['depth'] = _rotate_image(entries[placement]['depth'], idx)
            if 'normal' in type_map:
                entries[placement]['normal'] = _rotate_normal_image(entries[placement]['normal'], idx)
        return entries

    
