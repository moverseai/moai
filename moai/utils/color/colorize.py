import functools

import numpy as np
import torch
from matplotlib import colormaps as cm
from matplotlib.colors import Colormap

# import moai.utils.color.turbo @NOTE check if colormap turbo works with Matplotlib >= 3.3.0

__all__ = ["get_colormap", "get_color", "COLORMAPS"]


def _matplotlib_colormap(colormap: Colormap, tensor: torch.Tensor) -> np.ndarray:
    data = tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.detach().numpy()
    return colormap(data).squeeze(1).transpose(0, 3, 1, 2)[:, :3, :, :]


# NOTE: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed two minor releases later. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap(obj)`` instead.
jet = functools.partial(_matplotlib_colormap, cm.get_cmap("jet"))
magma = functools.partial(_matplotlib_colormap, cm.get_cmap("magma"))
inferno = functools.partial(_matplotlib_colormap, cm.get_cmap("inferno"))
plasma = functools.partial(_matplotlib_colormap, cm.get_cmap("plasma"))
seismic = functools.partial(_matplotlib_colormap, cm.get_cmap("seismic"))
viridis = functools.partial(_matplotlib_colormap, cm.get_cmap("viridis"))
viridis_r = functools.partial(_matplotlib_colormap, cm.get_cmap("viridis_r"))
gray = functools.partial(_matplotlib_colormap, cm.get_cmap("gray"))
cividis = functools.partial(_matplotlib_colormap, cm.get_cmap("cividis"))
bone = functools.partial(_matplotlib_colormap, cm.get_cmap("bone"))
bone_r = functools.partial(_matplotlib_colormap, cm.get_cmap("bone_r"))
turbo = functools.partial(_matplotlib_colormap, cm.get_cmap("turbo"))
turbo_r = functools.partial(_matplotlib_colormap, cm.get_cmap("turbo_r"))
binary = functools.partial(_matplotlib_colormap, cm.get_cmap("binary"))
raw_jet = cm.get_cmap("jet")
raw_magma = cm.get_cmap("magma")
raw_inferno = cm.get_cmap("inferno")
raw_plasma = cm.get_cmap("plasma")
raw_seismic = cm.get_cmap("seismic")
raw_turbo = cm.get_cmap("turbo")
raw_turbo_r = cm.get_cmap("turbo_r")
raw_gray = cm.get_cmap("gray")
raw_cividis = cm.get_cmap("cividis")
raw_bone = cm.get_cmap("bone")
raw_bone_r = cm.get_cmap("bone_r")
raw_viridis = cm.get_cmap("viridis")
raw_viridis_r = cm.get_cmap("viridis_r")
raw_binary = cm.get_cmap("binary")

COLORMAPS = {
    "jet": jet,
    "magma": magma,
    "inferno": inferno,
    "plasma": plasma,
    "seismic": seismic,
    "viridis": viridis,
    "viridis_r": viridis_r,
    "gray": gray,
    "cividis": cividis,
    "turbo": turbo,
    "turbo_r": turbo_r,
    "bone": bone,
    "bone_r": bone_r,
    "binary": binary,
}

RAW_COLORMAPS = {
    "jet": raw_jet,
    "magma": raw_magma,
    "inferno": raw_inferno,
    "plasma": raw_plasma,
    "seismic": raw_seismic,
    "viridis": raw_viridis,
    "viridis_r": raw_viridis_r,
    "gray": raw_gray,
    "cividis": raw_cividis,
    "turbo": raw_turbo,
    "turbo_r": raw_turbo_r,
    "bone": raw_bone,
    "bone_r": raw_bone_r,
    "binary": raw_binary,
}


def get_colormap(name: str) -> Colormap:
    return COLORMAPS[name] if name in COLORMAPS.keys() else COLORMAPS.items()[0][1]


def get_color(value: float, name: str = "turbo") -> np.array:
    colormap_data = RAW_COLORMAPS.get(name, raw_turbo)(range(256))[:, :3]
    length = len(colormap_data) - 1
    indexed_value = value * length
    lhs = np.clip(np.floor(value * length), 0, length)
    rhs = np.clip(np.ceil(value * length), 0, length)
    left = colormap_data[int(lhs)]
    right = colormap_data[int(rhs)]
    blend = rhs - indexed_value
    return left * blend + right * (1 - blend)


get_color_turbo = functools.partial(get_color, name="turbo")
get_color_turbo_r = functools.partial(get_color, name="turbo_r")
get_color_bone = functools.partial(get_color, name="bone")
get_color_bone_r = functools.partial(get_color, name="bone_r")
