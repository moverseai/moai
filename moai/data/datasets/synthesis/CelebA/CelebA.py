from moai.data.datasets.generic import StructuredImages

import functools

__all__ = ["CelebA"]

CelebA = functools.partial(StructuredImages, color={
    'glob': '*.jpg',
    'output_space': None,
})