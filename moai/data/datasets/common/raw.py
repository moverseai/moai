import torch
import typing
import toolz
import numpy as np
import pickle
import json

__all__ = [
    'load_npz_file',
    'load_pkl_file',
    'load_txt_file',
    'load_json_file',
]

def load_npz_file(
    filename:   str,
) -> typing.Dict[str, torch.Tensor]:
    data = np.load(filename)
    return toolz.valmap(lambda a: torch.from_numpy(a), data)

def load_pkl_file(
    filename:   str,
) -> typing.Dict[str, torch.Tensor]:
    data = { }
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_txt_file(
    filename:   str,
) -> typing.Dict[str, torch.Tensor]:
    data = np.loadtxt(filename)
    return data

def load_json_file(
    filename: str,
) -> typing.Dict[str, torch.Tensor]:
    data = {}
    with open(filename, 'rb') as f:
        data = json.load(f)
    return data