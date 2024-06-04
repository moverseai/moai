import logging

import torch

from moai.monads.generation.mesh.mesh_type import TriangleMesh
from moai.utils.arguments import assert_path

log = logging.getLogger(__name__)


class Vertices(torch.nn.Module):
    def __init__(self, filename: str):
        super(Vertices, self).__init__()
        assert_path(log, "mesh filename", filename)
        mesh = TriangleMesh.from_obj(filename)
        vertices = mesh.vertices.unsqueeze(0)
        faces = mesh.faces.int()  # TODO: support for faces too
        self.register_buffer("vertices", vertices.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.vertices.expand(b, *self.vertices.shape[1:])
