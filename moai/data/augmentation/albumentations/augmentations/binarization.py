import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

__all__ = ["Binarization"]


class Binarization(ImageOnlyTransform):
    def __init__(self, threshold: float = 0.85):
        super().__init__(always_apply=True, p=1.0)
        self.threshold = threshold

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return (img > self.threshold).astype(np.float32)
