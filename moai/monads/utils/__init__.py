import dataclasses

import torch

from moai.monads.utils.common import dim_list, expand_dims
from moai.monads.utils.spatial import (
    expand_spatial_dims,
    flatten_spatial_dims,
    spatial_dim_list,
    spatial_dims,
)

dataclass = dataclasses.dataclass(
    init=True, repr=True, eq=True, frozen=False, unsafe_hash=True
)


@dataclass
class DataModule(torch.nn.Module):
    def __post_init__(
        self,
    ) -> None:
        super().__init__()


__all__ = [
    "expand_dims",
    "dim_list",
    "spatial_dim_list",
    "expand_spatial_dims",
    "flatten_spatial_dims",
    "spatial_dims",
    "dataclass",
    "DataModule",
]
