import torch

__all__ = ["MeshVertexNormals", "MeshFaceNormals"]

# NOTE: From https://github.com/facebookresearch/pytorch3d/issues/736


class MeshVertexNormals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        # NOTE: only tested w/ shared faces tensor
        verts_normals = torch.zeros_like(vertices)
        faces = faces.squeeze()
        vertices_faces = vertices[:, faces]

        faces_normals = torch.cross(
            vertices_faces[..., 1, :] - vertices_faces[..., 0, :],
            vertices_faces[..., 2, :] - vertices_faces[..., 1, :],
            dim=-1,
        )
        # faces_normals = torch.nn.functional.normalize(faces_normals, dim=-1, eps=1e-8)

        verts_normals.index_add_(1, faces[..., 0], faces_normals)
        verts_normals.index_add_(1, faces[..., 1], faces_normals)
        verts_normals.index_add_(1, faces[..., 2], faces_normals)

        # verts_normals.index_add_(-2, faces[..., 0], faces_normals)

        # faces_normals = torch.cross(
        #     vertices_faces[..., 2] - vertices_faces[..., 1],
        #     vertices_faces[..., 0] - vertices_faces[..., 1],
        #     dim=-1,
        # )
        # verts_normals.index_add_(-2, faces[..., 1], faces_normals)

        # faces_normals = torch.cross(
        #     vertices_faces[..., 0] - vertices_faces[..., 2],
        #     vertices_faces[..., 1] - vertices_faces[..., 2],
        #     dim=-1,
        # )
        # verts_normals.index_add_(-2, faces[:, 2], faces_normals)

        return {
            "vectors": torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=-1),
            "areas": torch.abs(torch.linalg.norm(faces_normals, ord=2, dim=-1) * 0.5),
            # "areas": torch.abs(torch.linalg.norm(verts_normals, ord=2, dim=-1) * 0.5),
        }


class MeshFaceNormals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        # NOTE: only tested w/ shared faces tensor
        # verts_normals = torch.zeros_like(vertices)
        faces = faces.squeeze()
        vertices_faces = vertices[:, faces]

        faces_normals = torch.cross(
            vertices_faces[..., 1] - vertices_faces[..., 0],
            vertices_faces[..., 2] - vertices_faces[..., 1],
            dim=-1,
        )
        face_areas = torch.linalg.norm(faces_normals, ord=2, dim=-1) * 0.5
        return {
            "normals": torch.nn.functional.normalize(faces_normals, eps=1e-6, dim=-1),
            "areas": face_areas.abs(),
        }
