import logging
import typing

import torch
import torchvision

log = logging.getLogger(__name__)

__all__ = ["CIFAR10", "CIFAR100"]

# NOTE: normalization values taken from: https://github.com/kuangliu/pytorch-cifar/issues/19

__NORM_MEAN_STD__ = {
    "CIFAR10": {
        "mean": torch.Tensor([[[0.49139968]], [[0.48215841]], [[0.44653091]]]),
        "std": torch.Tensor([[[0.24703223]], [[0.24348513]], [[0.26158784]]]),
    },
    "CIFAR100": {
        "mean": torch.Tensor([[[0.50707516]], [[0.48654887]], [[0.44091784]]]),
        "std": torch.Tensor([[[0.26733429]], [[0.25643846]], [[0.27615047]]]),
    },
}


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root_path: str,
        split: str,
        download: bool = True,
    ):
        super(CIFAR10, self).__init__(
            root=root_path,
            train=split == "train",
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    # torchvision.transforms.RandomCrop(size=32, padding=4),
                    (
                        torchvision.transforms.RandomHorizontalFlip()
                        if split == "train"
                        else torchvision.transforms.Lambda(_identity)
                    ),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=__NORM_MEAN_STD__[__class__.__name__]["mean"]
                        .squeeze()
                        .tolist(),
                        std=__NORM_MEAN_STD__[__class__.__name__]["std"]
                        .squeeze()
                        .tolist(),
                        inplace=True,
                    ),
                ]
            ),
        )
        log.info(f"Loaded {len(self)} items from CIFAR10")

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        image, label = super(CIFAR10, self).__getitem__(index)
        return {
            "color": image,
            "label": label,
            "dataset_mean": __NORM_MEAN_STD__[__class__.__name__]["mean"],
            "dataset_std": __NORM_MEAN_STD__[__class__.__name__]["std"],
        }


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(
        self,
        root_path: str,
        split: str,
        download: bool = True,
    ):
        super(CIFAR100, self).__init__(
            root=root_path,
            train=split == "train",
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    # torchvision.transforms.RandomCrop(size=32, padding=4),
                    (
                        torchvision.transforms.RandomHorizontalFlip()
                        if split == "train"
                        else torchvision.transforms.Lambda(_identity)
                    ),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=__NORM_MEAN_STD__[__class__.__name__]["mean"]
                        .squeeze()
                        .tolist(),
                        std=__NORM_MEAN_STD__[__class__.__name__]["std"]
                        .squeeze()
                        .tolist(),
                        inplace=True,
                    ),
                ]
            ),
        )
        log.info(f"Loaded {len(self)} items from CIFAR100")

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        image, label = super(CIFAR100, self).__getitem__(index)
        return {
            "color": image,
            "label": label,
            "dataset_mean": __NORM_MEAN_STD__[__class__.__name__]["mean"],
            "dataset_std": __NORM_MEAN_STD__[__class__.__name__]["std"],
        }
