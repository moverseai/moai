import logging
import typing

import numpy as np

log = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None
    log.warning(
        "Please `pip install opencv-python` to use jet-colormap heatmap visualisation."
    )

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn("Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ["jet_heatmap", "heatmap_tensor"]


def _colorize(prob: np.ndarray, size: typing.Tuple[int, int]) -> np.ndarray:
    """prob: (h, w) non-negative -> (H, W, 3) RGB uint8, resized to `size` = (W, H) and mapped
    through jet, normalised per-map so a single confident peak is always fully saturated.
    """
    width, height = size
    normalized = prob / max(float(prob.max()), 1e-8)
    resized = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_LINEAR)
    as_u8 = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(as_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def jet_heatmap(
    heatmaps: np.ndarray,  # (B, K, h, w) or (K, h, w)
    path: str,
    names: typing.Optional[typing.Sequence[str]] = None,
    size: typing.Optional[typing.Tuple[int, int]] = None,  # (W, H); defaults to (w, h)
    max_keypoints: int = 8,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time("optimization_step", sequence=optimization_step)
    elif lightning_step is not None:
        rr.set_time("lightning_step", sequence=lightning_step)
    elif iteration is not None:
        rr.set_time("iteration", sequence=iteration)
    maps = heatmaps[0] if heatmaps.ndim == 4 else heatmaps  # first batch element only
    _, h, w = maps.shape
    out_size = size or (w, h)
    for index in range(min(max_keypoints, maps.shape[0])):
        label = names[index] if names is not None else str(index)
        rr.log(f"{path}/{label}", rr.Image(_colorize(maps[index], out_size)))


_TENSOR_VIEW_PATHS: typing.List[str] = []


def _send_tensor_views(paths: typing.Sequence[str], colormap: str) -> None:
    # tensor layout is (y, x, landmark): pin the image plane to y/x and the slider to landmarks
    # (the viewer's own default uses the first two dims). `auto_views` keeps the other views.
    views = [
        rrb.TensorView(
            origin=path,
            name=path.rsplit("/", 1)[-1],
            slice_selection=rrb.TensorSliceSelection(width=1, height=0, slider=[2]),
            scalar_mapping=rrb.TensorScalarMapping(colormap=colormap),
        )
        for path in paths
    ]
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views), auto_views=True))


def heatmap_tensor(
    heatmaps: np.ndarray,  # (B, K, h, w) or (K, h, w)
    path: str,
    dim_names: typing.Sequence[str] = ("y", "x", "landmark"),
    colormap: str = "turbo",
    blueprint: bool = True,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time("optimization_step", sequence=optimization_step)
    elif lightning_step is not None:
        rr.set_time("lightning_step", sequence=lightning_step)
    elif iteration is not None:
        rr.set_time("iteration", sequence=iteration)
    if blueprint and path not in _TENSOR_VIEW_PATHS:
        _TENSOR_VIEW_PATHS.append(path)
        _send_tensor_views(_TENSOR_VIEW_PATHS, colormap)
    maps = heatmaps[0] if heatmaps.ndim == 4 else heatmaps  # first batch element only
    rr.log(
        path, rr.Tensor(maps.transpose(1, 2, 0), dim_names=list(dim_names))
    )  # (y, x, K)
