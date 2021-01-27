from moai.utils.arguments import (
    assert_choices,
    assert_sequence_size
)

import torch
import functools
import itertools
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["FiniteDifference"]

class FiniteDifference(torch.nn.Module): #TODO: support for order
    
    class ReflectionPad3d(torch.nn.modules.padding._ReflectionPadNd):
        def __init__(self, padding: typing.Sequence[int]):
            super(FiniteDifference.ReflectionPad3d, self).__init__()
            self.padding = torch.nn.modules.utils._ntuple(6)(padding)

    __PAD_MAP__ = {
        "reflect": {
            1: torch.nn.ReflectionPad1d,
            2: torch.nn.ReflectionPad2d,
            3: FiniteDifference.ReflectionPad3d
        },
        "replicate": {
            1: torch.nn.ReplicationPad1d,
            2: torch.nn.ReplicationPad2d,
            3: torch.nn.ReplicationPad3d,
        },
        "zero": {
            1: functools.partial(torch.nn.ConstantPad1d, value=0.0),
            2: torch.nn.ZeroPad2d,
            3: functools.partial(torch.nn.ConstantPad3d, value=0.0),
        }
    }

    __PAD_TYPES__ = ['reflect', 'replicate', 'zero']
    __MODES__ = ['central', 'forward', 'backward']
    __REDUCTIONS__ = ['none', 'magnitude', 'absum']

    def __init__(self,
        dims:               typing.Sequence[int],
        mode:               str='central', # see __MODES__
        pad:                str='reflect', # see __PAD_TYPES__
        reduction:          str='magnitude', # see __REDUCTIONS__
        keep_size:          bool=False,
    ):
        super(FiniteDifference, self).__init__()
        assert_choices(log, "padding type", pad, FiniteDifference.__PAD_TYPES__)
        assert_choices(log, "mode", mode, FiniteDifference.__MODES__)
        assert_sequence_size(log, "dimensions", dims, max_size=3)
        self.min_dim = min(dims)
        self.scale = 0.5 if mode == 'central' else 1.0
        self.pad = self.get_padding(dims, pad, mode, keep_size)
        size_offset = 2 if mode == 'central' else 1
        nele_func = (lambda d: d) if keep_size else (lambda d: d - size_offset)
        self.nelements_func = lambda t: [nele_func(d) for d in t.shape]
        self.lhs_pad = torch.nn.Identity() if mode == 'backward' and not keep_size else self.pad
        self.rhs_pad = torch.nn.Identity() if mode == 'forward' and not keep_size else self.pad
        lhs_inds = 2 if mode == 'central' else 1
        other_ind = 0 if mode == 'forward' else 1
        self.lhs_narrow = [functools.partial(self.get_narrower, 
            dimensions=dims, dim_only=keep_size,
            this_index=lhs_inds, other_index=other_ind, dim=d,
        ) for d in dims]
        self.rhs_narrow = [functools.partial(self.get_narrower, 
            dimensions=dims, dim_only=keep_size,
            this_index=0, other_index=other_ind, dim=d,
        ) for d in dims]
        def abs_sum(t: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.sum(t.abs(), dim=[dim])
        self.reduction_func = functools.partial(abs_sum, dim=self.min_dim)\
            if reduction == 'abssum' else (
                functools.partial(torch.linalg.norm, p=2, dim=self.min_dim)
                if reduction == 'magnitude' else lambda t: t
            )

    def get_padding(self, 
        dims:               typing.Sequence[int],
        pad_type:           str,
        mode:               str,
        keep_size:          bool,
    ) -> torch.nn.Module:
        base_pad = [1, 1] if mode == 'central' else [0, 1]
        base_pad = list(reversed(base_pad)) if mode == 'backward' else base_pad
        padding = functools.reduce(lambda s, t: t + s, itertools.repeat(base_pad, len(dims)))
        pad_type = "ReplicationPad" # "ReflectionPad"
        pad = FiniteDifference.__PAD_MAP__[pad_type][len(dims)]\
            if keep_size else torch.nn.Identity
        return pad(padding)

    @staticmethod
    def get_narrower(
        tensor:             torch.Tensor,
        nelements:          typing.Sequence[int],
        dim:                int,
        this_index:         int,
        other_index:        int,        
        dimensions:         int,
        dim_only:           bool,
    ) -> typing.Callable:
        return functools.reduce(
            lambda t, i: torch.narrow(
                t, i, this_index if i == dim else other_index,
                nelements[i]
            ), dimensions, tensor
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        lhs_padded = self.lhs_pad(tensor)
        rhs_padded = self.rhs_pad(tensor)
        nelements = self.nelements_func(tensor)
        dTds = torch.stack([
            torch.sub(
                lhs(lhs_padded, nelements), rhs(rhs_padded, nelements)
            ) for lhs, rhs in zip(self.lhs_narrow, self.rhs_narrow)
        ], dim=self.min_dim)
        if self.scale != 1.0:
            dTds = self.scale * dTds
        return self.reduction_func(dTds)
