import albumentations.augmentations.crops.transforms as aug
import cv2

__all__ = ["RandomSizedCrop"]


# NOTE: wrapper as to circumvent hydra/tuple issue
class RandomSizedCrop(aug.RandomSizedCrop):
    def __init__(
        self,
        min_height: int,
        max_height: int,
        height: int,
        width: int,
        w2h_ratio: float,
        interpolation: cv2.INTER_LINEAR,
        always_apply: bool,
        p: float = 1.0,
    ):
        super(RandomSizedCrop, self).__init__(
            min_max_height=(min_height, max_height),
            height=height,
            width=width,
            w2h_ratio=w2h_ratio,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )
