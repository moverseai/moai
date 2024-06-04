import typing
from functools import partial

import toolz
import torch
from torch.utils.data.dataloader import default_collate


def _collate_fn(batch, ingore_keys):
    batch_ = []
    if isinstance(batch[0], dict):
        # remove keys
        for b in batch:
            d_ = toolz.dicttoolz.dissoc(b, *ingore_keys)
            batch_.append(d_)

    del batch
    return default_collate(batch_)


class Ignore(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        ignore_keys: typing.Sequence[str],
    ) -> None:
        super(Ignore, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=partial(_collate_fn, ingore_keys=ignore_keys),
        )
