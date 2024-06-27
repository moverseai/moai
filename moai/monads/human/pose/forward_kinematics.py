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
        offsets: typing.Optional[torch.Tensor] = None,  # [B, (T), J, 3]
        parents: typing.Optional[torch.Tensor] = None,  # [B, J]
    ) -> typing.Dict[str, torch.Tensor]:  # { [B, (T), J, 3], [B, (T), J, 3, 3] }
        offsets = (
            offsets[:, np.newaxis, ..., np.newaxis]
            if offsets is not None
            else self.offsets[:, np.newaxis, ..., np.newaxis]
        )  # NOTE: careful, col vs row major order
        transforms = torch.zeros(*rotation.shape[:-2], 4, 4, device=rotation.device)
        transforms[..., :3, :3] = rotation.clone()
        transforms[..., :3, 3] = offsets[..., 0].clone()
        transforms[..., 0, :3, 3] = position.clone()
        transforms[..., 3, 3] = 1.0
        # NOTE: currently the op does not support per batch item parents
        parent_indices = (
            parents[0].detach().cpu()
            if parents is not None
            else (self.parents[0].detach().cpu())
        )
        if (
            parent_indices.shape[-1] == offsets.shape[-3]
        ):  # NOTE: to support using the same parents everywhere
            parent_indices = parent_indices[1:]
        composed = [transforms[..., 0, :, :]]
        for current_idx, parent_idx in enumerate(
            parent_indices, start=1
        ):  # NOTE: assumes parents exclude root
            composed.append(
                torch.matmul(
                    composed[parent_idx],
                    transforms[..., current_idx, :, :],
                )
            )
        composed = torch.stack(composed, dim=-3)
        joints = composed[..., :3, 3]

        return {
            "positions": joints,
            "bone_transforms": composed,
        }
