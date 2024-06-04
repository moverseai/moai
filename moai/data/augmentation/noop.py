import logging

import torch

log = logging.getLogger(__name__)

__all__ = ["NoOp"]


class NoOp(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        super(NoOp, self).__init__()
        self.inner = dataset
        log.info(f"No data augmentation being used.")

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index: int):
        return self.inner[index]  # inner dataset item
