import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import toolz

__all__ = ["Concatenated"]

class Concatenated(torch.utils.data.Dataset):
    def __init__(self,
        datasets:       omegaconf.DictConfig,
        augmentation:   omegaconf.DictConfig=None,
        extracted_keys: typing.List[str]=[],
    ):
        super().__init__()
        self.datasets = []
        if augmentation is not None:
            for dataset in datasets.values():
                self.datasets.append(hyu.instantiate(
                    augmentation,
                    hyu.instantiate(dataset)
                ))
        else:
            from moai.data.augmentation import NoOp
            for dataset in datasets.values():
                self.datasets.append(NoOp(hyu.instantiate(dataset)))
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)
        self.keys = [k.split('.') for k in extracted_keys]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = self.dataset[index]
        if self.keys:            
            out = {}
            for k in self.keys:
                out = toolz.assoc_in(out, k, toolz.get_in(k, item))
            return out
        else:
            return item