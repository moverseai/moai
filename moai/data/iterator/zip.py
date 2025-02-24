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
        end: int,
        datasets: omegaconf.DictConfig,
        start: typing.Optional[int] = None,
        step: typing.Optional[int] = None,
        augmentation: omegaconf.DictConfig = None,
    ):
        super().__init__(datasets, augmentation)
        self.start = start or 0
        self.step = step or 1
        self.end = min(end, len(self.dataset))
        self.indices = list(range(self.start, self.end, self.step))

    def __len__(self) -> int:
        return min(len(self.indices), len(self.dataset))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return self.dataset[self.indices[index]]
