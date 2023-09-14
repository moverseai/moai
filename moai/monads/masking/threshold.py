import torch
import logging
import functools

log = logging.getLogger(__name__)

__all__ = ["Threshold"]

class Threshold(torch.nn.Module):
    CAST_OPS = {
        "float": lambda t: t.float(),
        "byte": lambda t: t.byte(),
    }

    def __init__(self,
        value: float,
        comparison: str="lower",
        dtype: str="float", # "byte" #TODO: change to 'mask' and 'weights' semantics?
    ):
        super(Threshold, self).__init__()
        self.threshold = value
        self.comp_op = torch.le if comparison == "lower"\
            else (
                torch.ge if comparison == "greater" else torch.ge
            ) #TODO: update properly (avoid implicit resolving to ge for all cases)
        if dtype not in Threshold.CAST_OPS:
            log.error("Casting operation type for Threshold monad should be either float or byte")
        self.cast_op = Threshold.CAST_OPS[dtype]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cast_op(self.comp_op(x, self.threshold))

LowerThan = functools.partial(Threshold, comparison='lower')
HigherThan = functools.partial(Threshold, comparison='greater')