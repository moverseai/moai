import logging
import typing
import numpy as np

log = logging.getLogger(__name__)

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache
    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ['posed_image']

def posed_image(
    image: np.ndarray, #NOTE: assumes batch_size=1, i.e. [C, H, W]
    path: str, 
    pose: typing.Optional[np.ndarray]=None, 
    optimization_step: typing.Optional[int]=None,
    step: typing.Optional[int]=None,
    iter: typing.Optional[int]=None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif step is not None:
        rr.set_time_sequence("step", step)
    elif iter is not None:
        rr.set_time_sequence("iter", iter)
    _, H, W = image.shape
    rr.log(path, rr.Pinhole(focal_length=5000, width=W, height=H))
    rr.log(path, rr.Image(image.transpose(-2, -1, -3)))
