import albumentations
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["OneOf"]

class OneOf(torch.utils.data.Dataset):
    def __init__(self,
        dataset:        torch.utils.data.Dataset,
        inputs:         typing.Sequence[str],
        outputs:        typing.Sequence[str],
        augmentations:  omegaconf.DictConfig={},
        probability:    float=0.5,
    ):
        super(OneOf, self).__init__()
        self.inner, self.inputs, self.outputs = dataset, inputs, outputs
        self.augmentations = []
        for k, v in augmentations.items():
            log.info(f"Using {k} augmentation /w a {v.p * 100.0}% probability.")
            self.augmentations.append(hyu.instantiate(v))
        self.oneof = albumentations.OneOf(self.augmentations, p=probability)
        weight_sum = sum(list(map(lambda a: a.p, self.augmentations)), 0.0)
        if weight_sum == 0.0:
            log.warning("OneOf weights are zero, reverting to a NoOp.")
            self.oneof = albumentations.OneOf([albumentations.NoOp()], p=1.0)
        elif weight_sum != 1.0:
            log.warning(f"OneOf augmentation weights do not sum up to unity ({weight_sum}), they will be normalized to unity.")

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = self.inner[index]
        for i, o in zip(self.inputs, self.outputs):
            data = {"image": item[i].numpy().transpose(1, 2, 0)} # numpy conversion overhead is insignificant
            augmented = self.oneof(**data)
            item[o] = torch.from_numpy(augmented["image"].transpose(2, 0, 1))
        return item