import logging
import typing

import numpy as np
import torch

__all__ = ["ForwardKinematics"]


log = logging.getLogger(__name__)


class ForwardKinematics(torch.nn.Module):
    def __init__(
        self,
        parents: typing.Optional[typing.Sequence[int]] = None,
        offsets: typing.Optional[typing.Sequence[typing.Sequence[float]]] = None,
    ):  # TODO: add a col/row major param to adjust offset slicing
        super().__init__()
        if parents is not None:
            self.register_buffer("parents", torch.Tensor(parents).int())
        if offsets is not None:
            log.warning("offset not set, should be passed in the forward method")
            self.register_buffer("offsets", torch.Tensor(offsets))

    def forward(
        self,  # TODO: add parents tensor input?
        rotation: torch.Tensor,  # [B, (T), J, 3, 3]
        position: torch.Tensor,  # [B, (T), 3]
        offset: typing.Optional[torch.Tensor] = None,  # [B, (T), J, 3]
        parents: typing.Optional[torch.Tensor] = None,  # [B, J]
    ) -> typing.Dict[str, torch.Tensor]:  # { [B, (T), J, 3], [B, (T), J, 3, 3] }
        joints = torch.empty(rotation.shape[:-1], device=rotation.device)
        joints[..., 0, :] = position.clone()  # first joint according to global position
        offset = (
            offset[:, np.newaxis, ..., np.newaxis]
            if offset is not None
            else self.offsets[:, np.newaxis, ..., np.newaxis]
        )  # NOTE: careful, col vs row major order
        # offset = offset[np.newaxis, :, np.newaxis, :] #NOTE: careful, col vs row major order
        global_rotation = rotation.clone()
        # global_rotation = torch.empty(rotation.shape, device=rotation.device)
        # global_rotation[..., 0, :3, :3] = rotation[..., 0, :3, :3].clone()
        # NOTE: currently the op does not support per batch item parents
        parent_indices = (
            parents[0].detach().cpu()
            if parents is not None
            else (self.parents[0].detach().cpu())
        )
        if (
            parent_indices.shape[-1] == offset.shape[-3]
        ):  # NOTE: to support using the same parents everywhere
            parent_indices = parent_indices[1:]
        for current_idx, parent_idx in enumerate(
            parent_indices, start=1
        ):  # NOTE: assumes parents exclude root
            joints[..., current_idx, :] = torch.matmul(
                global_rotation[..., parent_idx, :, :], offset[..., current_idx, :, :]
            ).squeeze(-1)
            global_rotation[..., current_idx, :, :] = torch.matmul(
                global_rotation[..., parent_idx, :, :].clone(),
                rotation[..., current_idx, :, :].clone(),
            )
            joints[..., current_idx, :] += joints[..., parent_idx, :]

        return {
            "positions": joints,
            "rotations": global_rotation,
        }
