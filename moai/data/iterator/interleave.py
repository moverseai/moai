from posixpath import lexists
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import toolz
import numpy as np
import logging

log = logging.getLogger(__name__)

__all__ = ["Interleaved"]

class Interleaved(torch.utils.data.Dataset):
    r"""Dataset creation by sampling from multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): DictConfig of datasets to be concatenated
        probabilities (sequence): List of probabilities to create sampling
        size (int): Output size of concatenated dataset
    """
    def __init__(self,
        datasets:       omegaconf.DictConfig,
        probabilities: typing.List[float],
        size: int,
        augmentation:   omegaconf.DictConfig=None,
        extracted_keys: typing.List[str]=[],
    ):
        super().__init__()
        self.datasets = []
        if len(datasets) != len(probabilities):
            if len(probabilities) == 0:
                log.warning(f"Probabilities have not been assigned."
                "To match the size of datasets, the probabilities will be automatically filled equally for each dataset.")
                probabilities = np.resize(1/len(datasets),len(datasets))
            else:
                log.warning(f"Less probabilities values were given ({probabilities}) than the operation supports."
                    "To match the size of datasets, the probabilities will be automatically filled with repeated copies of the first element.")
                probabilities = np.resize(np.array(probabilities),len(datasets))
        if np.sum(probabilities) != 1.0:
            log.warning(f"Probabilities do not sum up to unity ({np.sum(probabilities)}), they will be normalized to unity.")
            probabilities /= np.sum(probabilities)
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
        self.lengths = [len(dset) for dset in self.datasets]
        self.indices = np.random.choice(len(datasets), size = np.sum(self.lengths), p = probabilities)
        self.keys = [k.split('.') for k in extracted_keys]
        self.probabilities = probabilities
        self.size = size
        for d,p in zip(datasets.keys(),probabilities):
            log.info(f"Drawing samples from {d} with probability {'{:.2f}'.format(p * 100)}%")

    
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        dataset_idx = np.random.choice(len(self.datasets), p = self.probabilities) #sample from dataset with probabilities
        index = np.random.randint(0,self.lengths[dataset_idx]) 
        item = self.datasets[dataset_idx][index] 
        if self.keys:            
            out = {}
            for k in self.keys:
                out = toolz.assoc_in(out, k, toolz.get_in(k, item))
            return out
        else:
            return item