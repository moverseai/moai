import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

__all__ = ["Erosion"]


class Erosion(ImageOnlyTransform):
    def __init__(
        self,
        kernel_size: int = 3,
        iterations: int = 1,
    ):
        super().__init__(always_apply=True, p=1.0)
        self.ksize = (kernel_size, kernel_size)
        self.iters = iterations

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, self.ksize, (-1, -1))
        eroded = cv2.erode(
            img, element, iterations=self.iters, borderType=cv2.BORDER_CONSTANT
        )
        return eroded[..., np.newaxis] if len(eroded.shape) == 2 else eroded
