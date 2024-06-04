import logging
import typing

import hydra.utils as hyu
import numpy as np
import omegaconf.omegaconf
import toolz
import torch

log = logging.getLogger(__name__)

__all__ = ["Interleaved"]


class Interleaved(torch.utils.data.Dataset):
    r"""Dataset creation by sampling from multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): DictConfig of datasets to be concatenated
        probabilities (sequence): DictConfig of probabilities to create sampling
        size (int): Output size of concatenated dataset
    """

    def __init__(
        self,
        datasets: omegaconf.DictConfig,
        size: int,
        probabilities: omegaconf.DictConfig,  # typing.List[float]=[],
        augmentation: omegaconf.DictConfig = None,
        extracted_keys: typing.List[str] = [],
    ):
        super().__init__()
        self.datasets = {}

        zero_probability_keys = list(
            toolz.valfilter(lambda p: p == 0.0, probabilities).keys()
        )
        if len(zero_probability_keys):
            datasets = toolz.dissoc(datasets, *zero_probability_keys)
            probabilities = toolz.dissoc(probabilities, *zero_probability_keys)
            log.warning(
                f"Datasets ({zero_probability_keys}) with 0 probability will be ingored for sampling."
            )

        missing_datasets = set(datasets.keys()) - set(probabilities.keys())

        for key in missing_datasets:
            datasets = toolz.dissoc(datasets, key)
            log.warning(
                f"No probability has been assigned for dataset {key}"
                " and thus will be ingored for sampling."
            )

        if np.sum(list(probabilities.values())) != 1.0:
            log.warning(
                f"Probabilities do not sum up to unity ({np.sum(list(probabilities.values()))}), they will be normalized to unity."
            )
            probabilities = toolz.valmap(
                lambda x: x / np.sum(list(probabilities.values())), probabilities
            )
        if augmentation is not None:
            for key, dataset in zip(datasets.keys(), datasets.values()):
                self.datasets[key] = hyu.instantiate(
                    augmentation, hyu.instantiate(dataset)
                )
        else:
            from moai.data.augmentation import NoOp

            for key, dataset in zip(datasets.keys(), datasets.values()):
                self.datasets[key] = NoOp(hyu.instantiate(dataset))
        self.lengths = toolz.valmap(lambda x: x.__len__(), self.datasets)
        self.keys = [k.split(".") for k in extracted_keys]
        self.probabilities = probabilities
        self.size = size
        for d, p in probabilities.items():
            log.info(
                f"Drawing samples from {d} with probability {'{:.2f}'.format(p * 100)}%"
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        dataset_idx = np.random.choice(
            list(self.probabilities.keys()), p=list(self.probabilities.values())
        )  # sample from dataset with probabilities
        index = np.random.randint(0, self.lengths[dataset_idx])
        item = self.datasets[dataset_idx][index]
        if self.keys:
            out = {}
            for k in self.keys:
                out = toolz.assoc_in(out, k, toolz.get_in(k, item))
            return out
        else:
            return item
