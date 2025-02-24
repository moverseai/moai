import logging
import os
import typing

import numpy as np
import torch
import trimesh
from pytorch_lightning.callbacks import Callback

from moai.utils.arguments import ensure_choices, ensure_path

__all__ = ["Mesh"]

log = logging.getLogger(__name__)


class Mesh(
    Callback,
    typing.Callable[
        [typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]],
        None,
    ],
):
    __FORMATS__ = ["ply", "obj"]

    def __init__(
        self,
        path: str,
        extension: str = "ply",
        overwrite: bool = False,
        save_last: bool = False,
    ):
        self.path = (
            ensure_path(log, "output folder", path) if os.path.isdir(path) else path
        )
        self.format = ensure_choices(log, "output format", extension, Mesh.__FORMATS__)
        self.overwrite = overwrite

    def __call__(
        self,
        vertices: np.ndarray,  # TODO: add fps
        faces: np.ndarray,
        batch_idx: typing.Optional[int] = None,
        epoch: typing.Optional[int] = None,
    ) -> None:
        B = vertices.shape[0]
        assert len(vertices.shape) == 3 and len(faces.shape) >= 3
        if len(faces.shape) == 2:
            faces = faces.unsqueeze(0).expand(B, *faces.shape)
        for i, (v, f) in enumerate(zip(vertices, faces)):
            trimesh.Trimesh(v, f).export(f"mesh_e{epoch}_b{batch_idx}.{self.format}")
