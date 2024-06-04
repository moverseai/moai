import logging

import torch

from moai.utils.arguments import assert_choices

log = logging.getLogger(__name__)

__all__ = ["Finite"]


class Finite(torch.nn.Module):
    __CAST_OPS__ = {
        "float": lambda t: t.float(),
        "byte": lambda t: t.byte(),
        "bool": lambda t: t.bool(),
    }

    __MODES__ = ["nan", "inf", "nan+inf", "finite"]

    def __init__(
        self,
        mode: str = "finite",  # one of ['nan', 'inf', 'nan+inf', 'finite']
        dtype: str = "float",  # "byte" #TODO: change to 'mask' and 'weights' semantics?
    ):
        super(Finite, self).__init__()
        assert_choices(log, "mode", mode, Finite.__MODES__)
        assert_choices(log, "cast type", dtype, Finite.__CAST_OPS__.keys())
        self.check_func = (
            torch.isnan
            if mode == "nan"
            else (
                torch.isinf
                if mode == "inf"
                else (
                    torch.isfinite
                    if mode == "finite"
                    else lambda x: torch.isnan(x) & torch.isinf(x)
                )
            )
        )
        self.cast_op = Finite.__CAST_OPS__[dtype]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cast_op(self.check_func(x).all(dim=1, keepdim=True))
