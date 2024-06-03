import torch
import typing
import logging
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["GroundLoss"]


class SimpleGroundLoss(torch.nn.Module):
    def __init__(
        self,
        amp: float = 1000,
    ) -> None:
        super().__init__()
        self.amp = amp

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        # penalize vertices that are below the ground plane
        # get vertices under the ground plane
        ground_plane_height = 0.0
        vertex_height = (
            vertices[:, :, 1]
            - ground_plane_height  # assuming vertices are aligned along y axis
        )
        # penalize vertices that are below the ground plane
        # vertices above the ground plane are not penalized having a zero loss
        # vertices below the ground plane are penalized with a loss that is proportional to the distance from the ground plane
        # calculate the distance from the ground plane

        return torch.where(
            vertex_height < 0,
            self.amp * vertex_height**2,
            torch.zeros_like(vertex_height),
        )


class GroundLoss(torch.nn.Module):
    def __init__(
        self,
        out_alpha1: int = 1,
        out_alpha2: float = 0.15,
        in_alpha1: int = 10,
        in_alpha2: float = 0.15,
        hd_mesh_path: str = "smpl",
    ):
        super(GroundLoss, self).__init__()
        self.a1 = out_alpha1
        self.a2 = out_alpha2
        self.b1 = in_alpha1
        self.b2 = in_alpha2
        hd_operator_path = hd_mesh_path
        hd_operator = np.load(hd_operator_path)
        self.hd_operator = torch.sparse.FloatTensor(
            torch.tensor(hd_operator['index_row_col']),
            torch.tensor(hd_operator['values']),
            torch.Size(hd_operator['size']))
    
    
    @staticmethod
    def sparse_batch_mm(m1, m2):
        """
        https://github.com/pytorch/pytorch/issues/14489

        m1: sparse matrix of size N x M
        m2: dense matrix of size B x M x K
        returns m1@m2 matrix of size B x N x K
        """

        batch_size = m2.shape[0]
        # stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
        m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
        result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
            .transpose(1, 0)
        return result
        
    def _hdfy_mesh(self, vertices):
        """
        Applies a regressor that maps SMPL vertices to uniformly distributed vertices
        """
        # device = body.vertices.device
        # check if vertices ndim are 3, if not , add a new axis
        if vertices.ndim != 3:
            # batchify the vertices
            vertices = vertices[None, :, :]

        # check if vertices are an ndarry, if yes, make pytorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(self.device)

        vertices = vertices.to(torch.double)

        if self.hd_operator.device != vertices.device:
            self.hd_operator = self.hd_operator.to(vertices.device)
        hd_verts = self.sparse_batch_mm(self.hd_operator, vertices).to(torch.float)
        return hd_verts

    def forward(
        self,
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss that ensures that the ground plane pulls body mesh vertices towards it
        till contact and resolves ground-plane intersections
        Using assymetric loss such that the
           - loss is 0 when the body is in contact with the ground plane
           - loss is >> 0 when the plane interpenetrates the body (completely implausible)
           - loss >0 if the plane is close to nearby vertices that are not touching
        vertices: the vertices should be aligned along y axis and in world coordinates
        """
        # get vertices under the ground plane
        vertices_hd = self._hdfy_mesh(vertices)
        ground_plane_height = 0.0  # obtained by visualization on the presented pose
        vertex_height = (
            vertices_hd[:, :, 1]
            - ground_plane_height  # assuming vertices are aligned along y axis
        )
        inside_mask = vertex_height < 0.00
        outside_mask = vertex_height >= 0.00
        # pull closeby outside vertices
        v2v_pull = self.a1 * torch.tanh((vertex_height * outside_mask) / self.a2) ** 2
        # # apply loss to inside vertices to remove intersection
        v2v_push = self.b1 * torch.tanh((vertex_height * inside_mask) / self.b2) ** 2

        return v2v_pull + v2v_push
    

def ea2rm(x, y, z):
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)

    R = torch.stack(
            [torch.cat([cos_y*cos_z, sin_x*sin_y*cos_z - cos_x*sin_z, cos_x*sin_y*cos_z + sin_x*sin_z], dim=1),
            torch.cat([cos_y*sin_z, sin_x*sin_y*sin_z + cos_x*cos_z, cos_x*sin_y*sin_z - sin_x*cos_z], dim=1),
            torch.cat([-sin_y, sin_x*cos_y, cos_x*cos_y], dim=1)], dim=1)
    return R


if __name__ == "__main__":
    import pyarrow.parquet as pq
    import os
    import open3d as o3d
    

    # read base smpl mesh to get faces
    smpl_template = o3d.io.read_triangle_mesh(
        r"\\192.168.1.3\Public\Personal_spaces\Giorgos\smpl_male.ply"
    )
    output_dir = r"\\192.168.1.3\Public\Shared\recordings\tofis-various-MB-recordings-Jan-24\b5b68b4e-468b-48e0-9f03-3096a21295a0"
    # get vertices from a pq file
    pq_file = r"\\192.168.1.3\Public\Shared\recordings\tofis-various-MB-recordings-Jan-24\b5b68b4e-468b-48e0-9f03-3096a21295a0\bundle_gd.parquet"
    parquet_table = pq.read_table(pq_file).to_pandas()
    gd_loss = GroundLoss(
        hd_mesh_path = r"D:\repos\selfcontact\selfcontact-essentials\hd_model\smpl\smpl_neutral_hd_vert_regressor_sparse.npz"
    )
    simple_loss = SimpleGroundLoss()
    # vertices = torch.rand(50, 6890, 3)
    vertices = torch.Tensor(parquet_table["body_vertices"])
    # loss = simple_loss(vertices)
    R1 = ea2rm(torch.tensor([[np.radians(270)]]), torch.tensor([[np.radians(0)]]),
                   torch.tensor([[np.radians(0)]])).float().to(vertices.device).expand(vertices.shape[0], 3, 3)
    pred_vertices_world = torch.einsum('bki,bji->bjk', [R1, vertices])
    # loss = gd_loss(pred_vertices_world)
    loss = gd_loss(vertices)
    print(loss)
    # clamp loss for visualization purposes
    # loss = torch.clamp(loss, 0, 0.4)
    # create a color map for the vertices
    import matplotlib.pyplot as plt
    from matplotlib import cm

    cmap = plt.cm.get_cmap("jet")
    # apply the color map to the vertices
    colors = cmap(loss.detach().numpy())
    # create a mesh and apply color map to vertices
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector((vertices[0]))
    mesh.triangles = smpl_template.triangles
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[0, :, :3])
    # save mesh
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "gd_loss_ip.ply"), mesh)
