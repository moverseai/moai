import torchvision
import typing
import torch
import logging

log = logging.getLogger(__name__)

__all__ = ['CIFAR10', 'CIFAR100']

def _identity(x: torch.Tensor) -> torch.Tensor:
    return x

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,
        root_path:      str,
        split:          str,
        download:       bool=True,
    ):
        super(CIFAR10, self).__init__(
            root=root_path,
            train=split == 'train',
            download=download,
            transform=torchvision.transforms.Compose([
                # torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomHorizontalFlip() if split == 'train'\
                    else torchvision.transforms.Lambda(_identity),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                    inplace=True,
                )
            ])
        )
        log.info(f"Loaded {len(self)} items from CIFAR10")

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        image, label = super(CIFAR10, self).__getitem__(index)
        return {
            'color': image,
            'label': label,
        }

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self,
        root_path:      str,
        split:          str,
        download:       bool=True,
    ):
        super(CIFAR100, self).__init__(
            root=root_path,
            train=split == 'train',
            download=download,
            transform=torchvision.transforms.Compose([
                # torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomHorizontalFlip() if split == 'train'\
                    else torchvision.transforms.Lambda(_identity),
                torchvision.transforms.ToTensor(),                
                torchvision.transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                    inplace=True,
                )
            ])
        )
        log.info(f"Loaded {len(self)} items from CIFAR100")

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        image, label = super(CIFAR100, self).__getitem__(index)
        return {
            'color': image,
            'label': label,
        }