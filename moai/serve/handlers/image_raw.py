from collections.abc import Callable
from torchvision import transforms


import typing
import torch

import io
from PIL import Image

__all__ = ['ImageRaw']

class ImageRaw(Callable):
    def __init__(self) -> None:
        super().__init__()
        self.image_processing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> torch.Tensor:
        raw = io.BytesIO(data)

        image = Image.open(raw)
        image = self.image_processing(image).unsqueeze(0).to(device)
        return {'color': image}