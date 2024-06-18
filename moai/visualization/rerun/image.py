import logging
import typing

import colour
import numpy as np

from moai.engine.modules.rerun import Rerun

log = logging.getLogger(__name__)

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = [
    "posed_image",
    "keypoints",
    "multicam_posed_image",
]


def multicam_posed_image(
    images: np.ndarray,  # assumes num_cams x [C, H, W]
    path: str,
    poses: np.ndarray = None,  # assumes num_cams x [4, 4];
    intrinsics: np.ndarray = None,  # assumes num_cams x [3, 3];
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    # NOTE: DEBUG
    # images = images[0, :, 0, :, :]
    # poses = poses[:, 0, :, :]
    # intrinsics = intrinsics[:, 0, :, :]
    # NOTE: DEBUG
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    # get number of cameras
    num_cams = images.shape[0]
    _, __, H, W = images.shape
    for i in range(num_cams):
        rr.log(
            path + f"/cam_{i}",
            rr.Transform3D(
                translation=poses[i][:3, 3],
                mat3x3=poses[i][:3, :3],
                from_parent=True,
            ),
        )
        rr.log(
            path + f"/cam_{i}",
            rr.Pinhole(
                image_from_camera=intrinsics[i],
            ),
        )
        # log image
        rr.log(
            path + f"/cam_{i}",
            rr.Image(images[i].transpose(-1, -2, 0)),
        )


def posed_image(
    image: np.ndarray,  # NOTE: assumes batch_size=1, i.e. [C, H, W]
    path: str,
    pose: typing.Optional[np.ndarray] = None,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    _, __, H, W = image.shape
    rr.log(path, rr.Pinhole(focal_length=5000, width=W, height=H))
    rr.log(path, rr.Image(image.transpose(0, -2, -1, -3)))


def multicam_keypoints(
    keypoints: np.ndarray,
    path: str,
    color: str,
    confidence: np.ndarray,
    skeleton: typing.Optional[str] = None,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
):
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    # use confidence to get
    color = colour.Color(color)
    num_cams = keypoints.shape[0]
    for i in range(num_cams):
        path_parts = path.split("/")
        path_parts.insert(-1, f"cam_{i}")
        fp = "/".join(path_parts)
        if skeleton is None:
            rr.log(
                # path + f"/cam_{i}",
                fp,
                rr.Points2D(
                    positions=keypoints[i],
                    colors=np.tile(
                        np.array(color.get_rgb() + (1,)), (keypoints[i].shape[0], 1)
                    ),
                    # colors=plt.colormaps["hot"](confidence[0][i]),
                    radii=confidence[0][i] * 5,
                ),
            )


def keypoints(
    keypoints: np.ndarray,
    path: str,
    color: str,
    skeleton: typing.Optional[str] = None,
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)
    if skeleton is None:
        rr.log(
            path,
            rr.Points2D(
                positions=keypoints[0],
                colors=np.tile(
                    np.array(color.get_rgb() + (1,)), (keypoints.shape[1], 1)
                ),  # TODO: memoize))
            ),
        )
    else:
        class_id = Rerun.__LABEL_TO_CLASS_ID__[skeleton]
        rr.log(
            path,
            rr.Points2D(
                positions=keypoints[0],
                class_ids=class_id,
                labels=skeleton,
                keypoint_ids=np.arange(keypoints.shape[1]),
                colors=np.tile(
                    np.array(color.get_rgb() + (1,)), (keypoints.shape[1], 1)
                ),  # TODO: memoize))
            ),
        )
