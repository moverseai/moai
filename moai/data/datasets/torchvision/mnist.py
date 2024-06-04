import logging
import typing

import torch
import torchvision

log = logging.getLogger(__name__)


class MNIST(torchvision.datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: str,
        download: bool = True,
        # transform:      torchvision.transforms.Compose=None,
    ):
        super(MNIST, self).__init__(
            root=root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.1307],
                        std=[0.3081],
                        inplace=True,
                    ),
                ]
            ),
        )
        log.info(f"Loaded {len(self)} items from MNIST")

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        image, label = super(MNIST, self).__getitem__(index)
        return {
            "color": image,
            "label": label,
        }
