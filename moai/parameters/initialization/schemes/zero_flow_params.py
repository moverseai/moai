import logging
import typing

import torch

from moai.utils.torch import get_parameter

log = logging.getLogger(__name__)

__all__ = ["ZeroFlowParams"]


class ZeroFlowParams(typing.Callable[[torch.nn.Module], None]):
    def __init__(
        self,
        keys: typing.Sequence[str],
    ):
        self.keys = keys

    def __call__(self, module: torch.nn.Module) -> None:
        zeroed_keys = []
        for key in self.keys:
            try:
                m = get_parameter(module.named_flows, key)
                if m is not None:
                    with torch.no_grad():  # TODO: remove this and add in root apply call
                        m.zero_()
                        m.grad = None
                zeroed_keys.append(key)
            except:
                break
        all_zeroed_keys = ",".join(zeroed_keys)
        log.info(f"Zeroing out parameters: [cyan italic]\[{all_zeroed_keys}][/].")
