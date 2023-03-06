import functools
import torch
import numpy as np
from matplotlib import cm
from matplotlib.colors import Colormap

# import moai.utils.color.turbo @NOTE check if colormap turbo works with Matplotlib >= 3.3.0

__all__ = [
    "get_colormap",
    "COLORMAPS"
]

def _matplotlib_colormap(
    colormap: Colormap,
    tensor: torch.Tensor
) -> np.ndarray:
    data = tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.detach().numpy()
    return colormap(data).squeeze(1).transpose(0, 3, 1, 2)[:, :3, :, :]

jet = functools.partial(_matplotlib_colormap, cm.get_cmap('jet'))
magma = functools.partial(_matplotlib_colormap, cm.get_cmap('magma'))
inferno = functools.partial(_matplotlib_colormap, cm.get_cmap('inferno'))
plasma = functools.partial(_matplotlib_colormap, cm.get_cmap('plasma'))
seismic = functools.partial(_matplotlib_colormap, cm.get_cmap('seismic'))
viridis = functools.partial(_matplotlib_colormap, cm.get_cmap('viridis'))
viridis_r = functools.partial(_matplotlib_colormap, cm.get_cmap('viridis_r'))
gray = functools.partial(_matplotlib_colormap, cm.get_cmap('gray'))
cividis = functools.partial(_matplotlib_colormap, cm.get_cmap('cividis'))
bone = functools.partial(_matplotlib_colormap, cm.get_cmap('bone'))
bone_r = functools.partial(_matplotlib_colormap, cm.get_cmap('bone_r'))
turbo = functools.partial(_matplotlib_colormap, cm.get_cmap('turbo'))
turbo_r = functools.partial(_matplotlib_colormap, cm.get_cmap('turbo_r'))

binary = cm.get_cmap('binary')

COLORMAPS = {
    'jet': jet,
    'magma': magma,
    'inferno': inferno,
    'plasma': plasma,
    'seismic': seismic,
    'viridis': viridis,
    'viridis_r': viridis_r,
    'gray': gray,
    'cividis': cividis,
    'turbo': turbo,
    'turbo_r': turbo_r,
    'bone': bone,
    'bone_r': bone_r,
    'binary': binary
}

def get_colormap(name:str) -> Colormap:
    return COLORMAPS[name] if name in COLORMAPS.keys() else COLORMAPS.items()[0][1]
<<<<<<< Updated upstream
=======

def get_color(value: float, name: str='turbo') -> np.array:
    colormap_data = COLORMAPS.get(name, turbo)(range(256))[:, :3]
    length = len(colormap_data) - 1
    indexed_value = value * length
    lhs = np.clip(np.floor(value * length), 0, length)
    rhs = np.clip(np.ceil(value * length), 0 , length)
    left = colormap_data[int(lhs)]
    right = colormap_data[int(rhs)]
    blend = rhs - indexed_value
    return left * blend + right * (1 - blend)

get_color_turbo = functools.partial(get_color, name='turbo')
get_color_turbo_r = functools.partial(get_color, name='turbo_r')
get_color_bone = functools.partial(get_color, name='bone')
get_color_bone_r = functools.partial(get_color, name='bone_r')


    
>>>>>>> Stashed changes
