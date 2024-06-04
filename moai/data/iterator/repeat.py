import logging

import hydra.utils as hyu
import omegaconf.omegaconf
import torch

log = logging.getLogger(__name__)

__all__ = ["Repeated"]


class Repeated(torch.utils.data.Dataset):
    r"""
    Repeats dataset values for a given number.

    Args:
        datasets (torch.utils.data.Dataset): dataset to be repeated.
        num_repeats (int): number of repeats.
    """

    def __init__(
        self,
        datasets: torch.utils.data.Dataset,
        num_repeats: int,
        augmentation: omegaconf.DictConfig = None,
    ):
        super(Repeated, self).__init__()
        self.num_repeats = num_repeats
        if augmentation is not None:
            self.dataset = hyu.instantiate(
                augmentation, hyu.instantiate(next(iter(datasets.values())))
            )
        else:
            from moai.data.augmentation import NoOp

            self.dataset = NoOp(hyu.instantiate(next(iter(datasets.values()))))

    def __len__(self) -> int:
        return len(self.dataset) * self.num_repeats

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]
