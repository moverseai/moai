from moai.utils.arguments import ensure_numeric_list
from moai.monads.utils import expand_spatial_dims

import torch
import typing

__all__ = ["Grid"]

def _create_inclusive(size: typing.List[int]) -> torch.Tensor:
    return torch.stack(
        torch.meshgrid(
            *[torch.linspace(0.0, 1.0 - (1.0 / dim), dim) for dim in size],
        )
    ).unsqueeze(0)

def _create_exclusive(size: typing.List[int]) -> torch.Tensor:    
    return torch.stack(
        torch.meshgrid(
            *[(torch.arange(dim) + 0.5) / dim for dim in size],
        )
    ).unsqueeze(0)

def _scale_to_size(grid: torch.Tensor, size: typing.List[int]) -> torch.Tensor:
    sizes = torch.Tensor([*size]).unsqueeze(0)
    sizes = expand_spatial_dims(sizes, grid)
    return grid * sizes

__MODES__ = ['ndc', 'coord', 'norm']

__MODE_CONVERTERS__ = {
    'ndc': lambda g, _: torch.addcmul(torch.scalar_tensor(-1.0), g, torch.scalar_tensor(2.0)), # from [0, 1] to [-1, 1]
    'coord': _scale_to_size, # from [0, 1] to [0, S] where S := { W, (H), (D) }
    'norm': lambda g, _: g, # passthrough, remains at [0, 1]
}

class Grid(torch.nn.Module):
    __XYZ__ = 'zyx'

    def __init__(self,
        mode:           str='ndc', # 'ndc', 'coord', 'norm'
        width:          int=256,
        height:         int=1,
        depth:          int=1,
        inclusive:      bool=True,
        order:          str='xyz', # order of coords
        persistent:     bool=True, # save and re-use the grid buffer
    ):
        super(Grid, self).__init__()
        size = [n for n in [depth, height, width] if n > 1]
        unit_grid = _create_inclusive(size) if inclusive\
            else _create_exclusive(size)
        grid = __MODE_CONVERTERS__[mode](unit_grid, size)
        order = order[:len(size)]
        size_diff = len(Grid.__XYZ__) - len(order)
        indices = [Grid.__XYZ__.index(c) - size_diff for c in order if c in Grid.__XYZ__]
        grid = torch.stack([
            grid[:, i, ...] for i in indices
        ], dim=1)
        self.register_buffer("grid", grid, persistent)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        b = tensor.shape[0]
        return self.grid.expand(b, *self.grid.shape[1:])

class MultiScaleGrid(torch.nn.Module):
    def __init__(self,
        mode:           str='ndc', # 'ndc', 'coord', 'norm'
        width:          typing.Sequence[int]=256,
        height:         typing.Sequence[int]=1,
        depth:          typing.Sequence[int]=1,
        inclusive:      bool=True,
        reverse:        bool=False,
    ):
        super(MultiScaleGrid, self).__init__()
        sizes = []
        for w, h, d in zip(
            ensure_numeric_list(width),
            ensure_numeric_list(height),
            ensure_numeric_list(depth)
        ):
            sizes += [n for n in [d, h, w] if n > 1]
        for i, size in enumerate(sizes):
            unit_grid = _create_inclusive(size) if inclusive\
                else _create_exclusive(size)
            grid = __MODE_CONVERTERS__[mode](unit_grid, size)
            grid = torch.flip(grid, dims=[1]) if not reverse else grid
            self.register_buffer("grid_" + str(i), grid)

    def forward(self, tensor: torch.Tensor) -> typing.List[torch.Tensor]:
        b = tensor.shape[0]
        grids = []
        i = 0
        while hasattr("grid_" + str(i)):
            grid = getattr("grid_" + str(i))
            grids.append(grid.expand(b, *grid.shape[1:]))
            i += 1
        return grids