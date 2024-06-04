import logging

import torch

log = logging.getLogger(__name__)

# NOTE: see https://github.com/Lightning-AI/pytorch-lightning/discussions/16698
# NOTE: see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

__all__ = ["Precision"]


class Precision(object):
    def __init__(
        self,
        precision: str = "highest",  # one of 'highest', 'high', 'medium'
    ):
        # Supports three settings:
        #  - highest: float32 matrix multiplications use the float32 datatype (24 mantissa bits)
        #  - high: float32 matrix multiplications either use the TensorFloat32 datatype (10 mantissa bits) or the float32 datatype
        #  - medium: medium: float32 matrix multiplications use the TensorFloat32 datatype (8 mantissa bits)

        log.info(f"Setting PyTorch precision to {precision}.")
        torch.set_float32_matmul_precision(precision)
