from moai.data.datasets.common.image2d import load_color_image
from collections.abc import Callable

import typing
import torch
import logging
import cv2

__all__ = [
    'ImageFileInput',
    'ImageFileOutput'
]

log = logging.getLogger(__name__)

class ImageFileInput(Callable):
    def __init__(self,
        input_key:          str='color',
        output_key:         str='color',
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> torch.Tensor:
        path = data.get(self.input_key)
        log.info(f"Loading image file [key: {self.input_key}] @ {path}")
        return {self.output_key: load_color_image(path).unsqueeze(0).to(device) }

class ImageFileOutput(Callable):
    def __init__(self,
        input_key:          str='color',
    ) -> None:
        super().__init__()
        self.input_key = input_key

    def __call__(self, 
        data:   typing.Mapping[str, torch.Tensor],
        json:   typing.Mapping[str, typing.Any],
    ) -> typing.Sequence[typing.Dict[str, torch.Tensor]]:
        image = data.get(self.input_key)
        image = image.detach().cpu().numpy()
        outs = []
        for img, kvp in zip(image, json):
            filename = kvp['body'].get(self.input_key)
            log.info(f"Saving tensor [key: {self.input_key}] @ {filename}")
            cv2.imwrite(filename, img.transpose(1, 2, 0))
            outs.append({ self.input_key: 'Success' }) #NOTE: Object type should be JSON serializable
        return outs