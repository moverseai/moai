import albumentations.augmentations.transforms as aug

__all__ = ["RandomGamma"]


# NOTE: wrapper as to circumvent hydra/tuple issue
class RandomGamma(aug.RandomGamma):
    def __init__(
        self, low: float = 80, high: float = 120, eps: float = 1e-7, p: float = 0.5
    ):
        super(RandomGamma, self).__init__(
            gamma_limit=(low, high), eps=eps, always_apply=False, p=p
        )
