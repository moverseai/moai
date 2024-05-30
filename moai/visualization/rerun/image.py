from moai.engine.modules.rerun import Rerun

import logging
import typing
import numpy as np
import colour

log = logging.getLogger(__name__)

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache
    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = [
    'posed_image',
    'keypoints',
]

def posed_image(
    image: np.ndarray, #NOTE: assumes batch_size=1, i.e. [C, H, W]
    path: str, 
    pose: typing.Optional[np.ndarray]=None, 
    optimization_step: typing.Optional[int]=None,
    lightning_step: typing.Optional[int]=None,
    iteration: typing.Optional[int]=None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    _, H, W = image.shape
    rr.log(path, rr.Pinhole(focal_length=5000, width=W, height=H))
    rr.log(path, rr.Image(image.transpose(-2, -1, -3)))

def keypoints(
    keypoints: np.ndarray,
    path: str,
    color: str,
    skeleton: typing.Optional[str]=None,
    optimization_step: typing.Optional[int]=None,
    lightning_step: typing.Optional[int]=None,
    iteration: typing.Optional[int]=None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    if skeleton is None:
        rr.log(path, rr.Points2D(positions=keypoints, 
            colors=np.tile(np.array(color.get_rgb() + (1,)), (keypoints.shape[0], 1)) #TODO: memoize))
        ))
    else:
        class_id = Rerun.__LABEL_TO_CLASS_ID__[skeleton]
        rr.log(path, rr.Points2D(positions=keypoints, class_ids=class_id,
            labels=skeleton, keypoint_ids=np.arange(keypoints.shape[0]),
            colors=np.tile(np.array(color.get_rgb() + (1,)), (keypoints.shape[0], 1)) #TODO: memoize))
        ))
