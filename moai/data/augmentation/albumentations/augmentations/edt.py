import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from scipy.ndimage import distance_transform_edt

__all__ = ["EDT"]

# NOTE: distance towards zero pixel


class EDT(ImageOnlyTransform):
    def __init__(
        self,
        symmetric: bool = False,
        inverse: bool = False,
        scale_inner: float = 1.0,
        scale_outer: float = 1.0,
    ):
        super().__init__(always_apply=True, p=1.0)
        self.symmetric = symmetric
        self.inverse = inverse
        self.scale_inner = scale_inner
        self.scale_outer = scale_outer

    def inverse_edt(self, img: np.ndarray) -> np.ndarray:
        binary = np.logical_not(img)
        return binary * distance_transform_edt(binary)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        b = img.astype(np.bool8)
        if self.inverse:
            b = np.logical_not(b)
        edt = distance_transform_edt(b)
        return (
            edt
            if not self.symmetric
            else b * edt * self.scale_inner + self.inverse_edt(b) * self.scale_outer
        )
