import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing

__all__ = ["Indexed"]

class Indexed(torch.utils.data.Dataset):
    def __init__(self,
        augmentation: omegaconf.DictConfig,
        datasets: omegaconf.DictConfig,
    ):
        super(Indexed, self).__init__()
        self.dataset = hyu.instantiate(
            augmentation,
            hyu.instantiate(next(iter(datasets.values())))
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return self.dataset[index]