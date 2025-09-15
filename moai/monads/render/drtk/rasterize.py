import functools
import logging

import torch

log = logging.getLogger(__name__)

try:
    import drtk
except ImportError:
    log.warning(
        "Failed to import drtk. Please install drtk from https://github.com/facebookresearch/DRTK"
    )


__all__ = ["Rasterize"]


class Rasterize(torch.nn.Module):
    def __init__(self, width: int, height: int, wireframe: bool = False):
        """ "

        Args:
            width (int): Width of the rasterized image.
            height (int): Height of the rasterized image.
            wireframe (bool): If True, rasterizes wireframes instead of filled triangles.

            vertices: (torch.Tensor): Tensor of shape (B, N, 3) representing the 3D vertices in image plane.
            faces: (torch.Tensor): Tensor of shape (B, M, 3) representing the triangle faces by indexing into vertices.

        Returns:
            The rasterized image of triangle indices which is represented with an index tensor of a shape [N, H, W].
        """
        super().__init__()
        self.width = width
        self.height = height
        self.wireframe = wireframe
        self.rasterizer = functools.partial(
            drtk.rasterize,
            width=self.width,
            height=self.height,
            wireframe=self.wireframe,
        )

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        # faces should be int32
        if faces.dtype != torch.int32:
            faces = faces.to(torch.int32)
        # NOTE: workaround for now to work with
        # vertices of shape: F, A, C, V, 3
        # faces of shape: F, A, T, 3
        masks_all = []
        # index_img = self.rasterizer(vertices, faces)
        # masks = index_img >= 0
        for a in range(vertices.shape[1]):
            for c in range(vertices.shape[2]):
                index_img = self.rasterizer(vertices[:, a, c], faces[:, a])
                masks = index_img >= 0
                masks_all.append(masks)
        return masks
