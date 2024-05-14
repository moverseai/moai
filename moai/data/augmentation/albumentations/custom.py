from moai.monads.execution._cascade import _create_accessor

import albumentations
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Custom"]

class Custom(torch.utils.data.Dataset):
    def __init__(self,
        dataset:        torch.utils.data.Dataset,
        inputs:         typing.Sequence[str],        
        outputs:        typing.Sequence[str],
        keys:           typing.Sequence[str],
        extra:          typing.List[typing.Mapping[str, str]]=[],
        augmentations:  omegaconf.DictConfig={},
    ):
        super().__init__()
        self.inner, self.inputs, self.outputs, self.keys, self.extra =\
            dataset, inputs, outputs, keys, extra
        # self.inputs = [_create_accessor(k) for k in self.inputs]
        self.inputs = [_create_accessor(k) if isinstance(k, str) else [_create_accessor(kk) for kk in k] for k in self.inputs]
        self.extra = [{k: _create_accessor(v) for k, v in (e or {}).items()} for e in self.extra]
        self.augmentations = []
        for k, aug in augmentations.items():
            # aug = hyu.instantiate(v)
            log.info(f"Using {k} augmentation /w a {aug.p * 100.0}% probability.")
            self.augmentations.append(aug)
        self.composition = albumentations.Compose(self.augmentations)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = self.inner[index]
        for i, o, k, e in zip(self.inputs, self.outputs, self.keys, self.extra):
            if isinstance(k, str): 
                data = {k: i(item).numpy() if isinstance(i(item),torch.Tensor) else i(item)}# numpy conversion overhead is insignificant
                for ek, ev in e.items():
                    data[ek] = ev(item).numpy() if isinstance(ev(item),torch.Tensor) \
                               else ev(item)
                augmented = self.composition(**data)
                if augmented[k] is not None:
                    item[o] = torch.from_numpy(augmented[k]) if isinstance(augmented[k],np.ndarray) else augmented[k] 
                    #item[o] = augmented[k] if isinstance(augmented[k],dict) else torch.from_numpy(augmented[k]) 
            else: # list
                data = {kk: i[j](item).numpy() if isinstance(i[j](item), torch.Tensor) else i[j](item) for j, kk in enumerate(k)}
                for ek, ev in e.items():
                    data[ek] = ev(item).numpy() if isinstance(ev(item),torch.Tensor) \
                             else ev(item)
                augmented = self.composition(**data)
                for kk, oo in zip(k, o):
                    if augmented[kk] is not None:
                        #item[oo] = torch.from_numpy(augmented[kk])
                        item[oo] = torch.from_numpy(augmented[kk])\
                            if isinstance(augmented[kk], np.ndarray) else augmented[kk]
                        # item[oo] = augmented[kk] if isinstance(augmented[kk],dict) else \
                        #     torch.from_numpy(augmented[kk])
        return item