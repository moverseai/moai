import torch
import typing
import toolz
import numpy as np


def load_npz_files(
    filename:   str,
) -> typing.Dict[str, torch.Tensor]:
    data = np.load(filename)
    return toolz.valmap(lambda a: torch.from_numpy(a), data)
