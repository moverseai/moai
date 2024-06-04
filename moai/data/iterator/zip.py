import logging
import typing

import hydra.utils as hyu
import omegaconf.omegaconf
import toolz
import torch

__all__ = ["Zipped", "SubsetZipped"]

logger = logging.getLogger(__name__)


class Zipper(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets=typing.Sequence[torch.utils.data.Dataset],
    ):
        self.datasets = datasets
        if len(list(toolz.unique(map(len, self.datasets)))) > 1:

            def _name_key(d: torch.utils.data.Dataset):
                return f"{d.__class__.__name__}: {len(d)}"

            logger.warning(
                f"Zipped datasets are of unequal lengths ("
                f"{list(map(_name_key, self.datasets))}),"
                f" will reduce length to smallest one ({min(map(len, self.datasets))})."
            )

    def __len__(self):
        return min(map(len, self.datasets))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        out = {}
        for d in self.datasets:
            out = toolz.merge(out, d[index])
        return out


class Zipped(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: omegaconf.DictConfig,
        augmentation: omegaconf.DictConfig = None,
    ):
        super(Zipped, self).__init__()
        if augmentation is not None:
            self.dataset = hyu.instantiate(
                augmentation, Zipper([hyu.instantiate(d) for d in datasets.values()])
            )
        else:
            from moai.data.augmentation import NoOp

            self.dataset = NoOp(Zipper([hyu.instantiate(d) for d in datasets.values()]))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return self.dataset[index]


class SubsetZipped(Zipped):
    def __init__(
        self,
        length: int,
        datasets: omegaconf.DictConfig,
        augmentation: omegaconf.DictConfig = None,
    ):
        super().__init__(datasets, augmentation)
        self.length = length

    def __len__(self) -> int:
        return min(self.length, len(self.dataset))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return self.dataset[index]
