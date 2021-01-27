from moai.monads.utils.common import (
    expand_dims,
    dim_list,
    dims
)

import torch
import functools
import typing

#TODO: use torch.flatten?
def flatten_spatial_dims(
    tensor:         torch.Tensor,
    spatial_start_index:    int=2,
) -> torch.Tensor:
    dims = [*tensor.shape[:spatial_start_index]] + [-1]
    return tensor.view(*dims)

spatial_dim_list = functools.partial(dim_list, start_index=2)
expand_spatial_dims = functools.partial(expand_dims, start_index=2)
spatial_dims = functools.partial(dims, start_index=2)
