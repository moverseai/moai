import logging
import os
import pickle
import typing

import mesh_intersection.loss as collisions_loss
import torch
from mesh_intersection.bvh_search_tree import BVH
from mesh_intersection.filter_faces import FilterFaces

# TODO: try/catch imports with error message


# NOTE: code from https://github.com/vchoutas/torch-mesh-isect

__all__ = ["Interpenetration"]

log = logging.getLogger(__name__)


class Interpenetration(torch.nn.Module):
    def __init__(
        self,
        max_collisions: int = 128,
        df_cone_height: float = 0.0001,
        penalize_outside: bool = True,
        point2plane: bool = False,
        part_segmentation: str = "",
        ignore_part_pairs: typing.Sequence[str] = [
            "9,16",
            "9,17",
            "6,16",
            "6,17",
            "1,2",
            "12,22",
        ],
    ):
        super(Interpenetration, self).__init__()
        if not torch.cuda.is_available():
            log.error(
                "The interpenetration loss cannot be used without any CUDA devices."
            )
        self.search_tree = BVH(max_collisions=max_collisions)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
            sigma=df_cone_height,
            point2plane=point2plane,
            vectorized=True,
            penalize_outside=penalize_outside,
        )
        if part_segmentation:
            if not os.path.exists(part_segmentation):
                log.error(
                    f"Part segmentation file ({part_segmentation}) given to interpenetration loss does not exist."
                )
            part_segmentation = os.path.expandvars(part_segmentation)
            with open(part_segmentation, "rb") as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file, encoding="latin1")
            faces_segm = face_segm_data["segm"]
            faces_parents = face_segm_data["parents"]
            self.filter_faces = (
                FilterFaces(  # Create the module used to filter invalid collision pairs
                    faces_segm=faces_segm,
                    faces_parents=faces_parents,
                    ign_part_pairs=ignore_part_pairs,
                )
            )
        else:
            self.filter_faces = torch.nn.Identity()

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        b = vertices.shape[0]
        triangles = torch.index_select(vertices, 1, faces.view(-1)).view(b, -1, 3, 3)
        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)
        collision_idxs = self.filter_faces(collision_idxs)  # Remove unwanted collisions
        return (
            self.pen_distance(triangles, collision_idxs)
            if collision_idxs.ge(0).sum().item() > 0
            else torch.scalar_tensor(0.0).to(vertices)
        )
