import logging
import typing

import numpy as np
import torch

from moai.utils.arguments import assert_positive

logger = logging.getLogger(__name__)

__all__ = ["WeakPerspective"]


class WeakPerspective(
    torch.nn.Module
):  # NOTE: fixed focal/principal, optimized rot/trans
    def __init__(
        self,
        focal_length: typing.Union[float, typing.Tuple[float, float]] = 5000.0,
        principal_point: typing.Union[float, typing.Tuple[float, float]] = 0.5,
        width: int = None,
        height: int = None,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
        persistent: bool = False,
    ):
        super(WeakPerspective, self).__init__()
        mat = torch.zeros([1, 2, 2])
        mat[:, 0, 0] = (
            focal_length if isinstance(focal_length, float) else focal_length[0]
        )
        mat[:, 1, 1] = (
            focal_length if isinstance(focal_length, float) else focal_length[1]
        )
        self.register_buffer("mat", mat, persistent=persistent)
        center = torch.zeros([1, 2])
        if principal_point is None:
            assert_positive(logger, "width", width)
            assert_positive(logger, "height", height)
            center[:, 0] = width // 2
            center[:, 1] = height // 2
        else:
            center[:, 0] = torch.scalar_tensor(
                principal_point
                if isinstance(principal_point, float)
                else principal_point[0]
            )
            center[:, 1] = torch.scalar_tensor(
                principal_point
                if isinstance(principal_point, float)
                else principal_point[1]
            )
        self.register_buffer("principal_point", center, persistent=persistent)
        if rotation is None:
            rotation = torch.eye(3).unsqueeze(dim=0)
        rotation = torch.nn.Parameter(rotation, requires_grad=True)
        self.register_parameter("rotation", rotation)
        if translation is None:
            translation = torch.zeros([1, 3])
        translation = torch.nn.Parameter(translation, requires_grad=True)
        self.register_parameter("translation", translation)

    def forward(
        self,
        points: torch.Tensor,
        image: torch.Tensor = None,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        # TODO: update with focal/principal inputs as well
    ) -> torch.Tensor:
        if image is not None:
            h, w = image.shape[-2:]
            with torch.no_grad():
                self.principal_point[:, 0] = w // 2
                self.principal_point[:, 1] = h // 2
        R = (
            rotation
            if rotation is not None
            else self.rotation.expand(points.shape[0], 3, 3)
        )
        t = (
            translation
            if translation is not None
            else self.translation.expand(points.shape[0], 3)
        )
        camera_transform = torch.cat(
            [
                torch.nn.functional.pad(R, [0, 0, 0, 1]),
                torch.nn.functional.pad(t.unsqueeze(dim=-1), [0, 0, 0, 1], value=1),
            ],
            dim=2,
        )
        z = torch.ones_like(points[..., :1])
        homogeneous_points = torch.cat([points, z], dim=-1)
        projected_points = torch.einsum(
            "bki,bji->bjk", [camera_transform, homogeneous_points]
        )
        img_points = torch.div(
            projected_points[:, :, :2], projected_points[:, :, 2].unsqueeze(dim=-1)
        )
        if intrinsics is not None:
            mat = intrinsics[:, :2, :2]
            principal_point = intrinsics[:, :2, 2]
        else:
            mat = self.mat
            principal_point = self.principal_point
        img_points = torch.einsum(
            "bki,bji->bjk", [mat, img_points]
        ) + principal_point.unsqueeze(
            dim=1
        )  # TODO: add principal in mat
        return img_points


class MVWeakPerspective(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        points: torch.Tensor,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        transform: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Multi-view weak perspective projection.

        # points [T,L,3]

        Expects tensors of  shape e.g. [T,N,L,3] where:
        T is the temporal window,
        N is the camera views,
        L is the number of points.
        """
        # points_cam = torch.einsum('bpi,bvij->bvpj', points, transform[:, :, :3, :3]) + transform[:, :, np.newaxis, :3, 3]
        points_cam = (
            torch.einsum(
                "bpoj,bvij->bpvi", points[..., None, :], transform[..., :3, :3]
            ).transpose(2, 1)
            + transform[..., None, :3, 3]
        )
        homo_points = points_cam / (points_cam[..., 2:3] + 1e-7)
        return torch.einsum("bvpi,bvji->bvpj", homo_points, intrinsics)[..., :2]


class MVWeakPerspectiveMultiActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        points: torch.Tensor,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        transform: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Multi-view weak perspective projection.

        Expects tensors of shape e.g. [B,A,L,3] where:
        B is batch size,
        A is the number of actors,
        L is the number of points.
        """
        # # 1. Rotate points
        # # points[..., None, :]  # gives [B, A, L, 1, 3]
        # # transform[..., :3, :3]  # gives [B, V, 3, 3]
        # # einsum output target: [B, A, V, L, 3] (for rotated points)
        rotated_points = torch.einsum(
            "baloj,bvij->bavli",
            points[..., None, :],  # b a l o j_coord
            transform[..., :3, :3],  # b v j_coord i_coord
        )  # Output shape: [B, A, V, L, 3] (b a v l i_coord)

        # 2. Add translation
        # translation_vector has shape [B, V, 3]
        translation_vector = transform[..., :3, 3]
        # Expand for broadcasting to [B, 1, V, 1, 3] to match [B, A, V, L, 3]
        # This adds a dimension for A (actors) and L (landmarks)
        translation_to_add = translation_vector[:, None, :, None, :]

        points_cam = rotated_points + translation_to_add  # Shape: [B, A, V, L, 3]

        # Convert to homogeneous coordinates by dividing by z
        homo_points = points_cam / (
            points_cam[..., 2:3] + 1e-7
        )  # Shape: [B, A, V, L, 3]

        # Project to image coordinates
        # intrinsics shape: [B, V, 3, 3]
        # homo_points shape: [B, A, V, L, 3]
        # Target uv_points shape: [B, A, V, L, 2]
        # We need to broadcast intrinsics over the A dimension or align dimensions carefully.
        # intrinsics[:, None, ...] gives [B, 1, V, 3, 3]
        uv_points = torch.einsum(
            "bavli,bavji->bavlj",  # contract i (coords)
            homo_points,
            intrinsics[:, None, ...],  # b (1) v j i -> effectively b a_broadcast v j i
        )[..., :2]

        return uv_points
