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
}

def get_colormap(name:str) -> Colormap:
    return COLORMAPS[name] if name in COLORMAPS.keys() else COLORMAPS.items()[0][1]


    