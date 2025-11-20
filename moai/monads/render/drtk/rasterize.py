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
        self.tensor = []
        self.rasterizer = functools.partial(
            drtk.rasterize,
            width=self.width,
            height=self.height,
            wireframe=self.wireframe,
        )

    def save_tensor(self, x: torch.Tensor) -> None:
        self.tensor.append(x)

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
            masks_per_actor = []
            for c in range(vertices.shape[2]):
                index_img = self.rasterizer(vertices[:, a, c], faces[:, a])
                masks = index_img >= 0
                depth_img, bary_img = drtk.render(
                    vertices[:, a, c], faces[:, a], index_img
                )
                image = (index_img != -1).float()
                if True:  # image.requires_grad:
                    image_differentiable = drtk.edge_grad_estimator(
                        vertices[:, a, c],
                        faces[:, a],
                        bary_img,
                        image[:, None],
                        index_img,
                        v_pix_img_hook=self.save_tensor,
                    )

                    masks_per_actor.append(
                        image_differentiable.squeeze(1)
                    )  # remove channel dim
                else:
                    masks_per_actor.append(masks)
            masks_all.append(torch.stack(masks_per_actor, dim=1))
        # return masks
        return torch.stack(masks_all, dim=1).to(vertices)  # .to(torch.float32)
