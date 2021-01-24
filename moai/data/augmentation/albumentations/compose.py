import albumentations
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Compose"]

class Compose(torch.utils.data.Dataset):
    def __init__(self,
        dataset:        torch.utils.data.Dataset,        
        inputs:         typing.Sequence[str],
        outputs:        typing.Sequence[str],
        augmentations:  omegaconf.DictConfig={},
    ):
        super(Compose, self).__init__()
        self.inner, self.inputs, self.outputs = dataset, inputs, outputs
        self.augmentations = []
        for k, v in augmentations.items():
            log.info(f"Using {k} augmentation /w a {v.p * 100.0}% probability.")
            self.augmentations.append(hyu.instantiate(v))
        self.composition = albumentations.Compose(self.augmentations)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = self.inner[index]
        for i, o in zip(self.inputs, self.outputs):
            data = {"image": item[i].numpy().transpose(1, 2, 0)} # numpy conversion overhead is insignificant
            augmented = self.composition(**data)
            item[o] = torch.from_numpy(augmented["image"].transpose(2, 0, 1))
        return item