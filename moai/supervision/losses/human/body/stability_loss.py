import torch
import numpy as np

class StabilityLossCoP(torch.nn.Module):
    def __init__(self,
                 faces,
                 cop_w = 10,
                 cop_k = 100,
                 contact_thresh:float = 0.1,
                 model_type='smpl',
                 device='cuda',
                 hd_mesh_path='smplx'
    ):
        super().__init__()
        """
        Loss that ensures that the COM of the SMPL mesh is close to the center of support 
        """
        if model_type == 'smpl':
            num_faces = 13776
            num_verts_hd = 20000

        assert faces is not None, 'Faces tensor is none'
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long).to(device)
        self.register_buffer('faces', faces)

        self.cop_w = cop_w
        self.cop_k = cop_k
        self.contact_thresh = contact_thresh

        with open(SMPL_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        with open(PART_VID_FID, 'rb') as f:
            self.part_vid_fid = pkl.load(f)

        # mapping between vid_hd and fid
        with open(HD_SMPL_MAP, 'rb') as f:
            faces_vert_is_sampled_from = pkl.load(f)['faces_vert_is_sampled_from']
        index_row_col = torch.stack(
            [torch.LongTensor(np.arange(0, num_verts_hd)), torch.LongTensor(faces_vert_is_sampled_from)], dim=0)
        values = torch.ones(num_verts_hd, dtype=torch.float)
        size = torch.Size([num_verts_hd, num_faces])
        hd_vert_on_fid = torch.sparse.FloatTensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, 'rb') as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros([len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.

        # mapping between vid_hd and part label
        self.hd_vid_in_part = self.vertex_id_to_part_mapping(hd_vert_on_fid, fid_to_part)
        
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
    
    def compute_triangle_area(self, triangles):
        ### Compute the area of each triangle in the mesh
        # Compute the cross product of the two vectors of each triangle
        # Then compute the length of the cross product
        # Finally, divide by 2 to get the area of each triangle

        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1,0).to(vertices.device)

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def forward(self, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self._hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        # # get COM of the SMPLX mesh
        # triangles = torch.index_select(vertices, 1, self.faces.view(-1)).reshape(batch_size, -1, 3, 3)
        # triangle_centroids = torch.mean(triangles, dim=2)
        # triangle_area = self.compute_triangle_area(triangles)
        # com_naive = torch.einsum('bij,bi->bj', triangle_centroids, triangle_area) / torch.sum(triangle_area, dim=1)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 1] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (1-self.cop_k*vertex_height) + outside_mask *  torch.exp(-self.cop_w * vertex_height)
        cop = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (torch.sum(pressure_weights, dim=1, keepdim=True) +eps)

        # naive center of support
        # vertex_height_robustified = GMoF_unscaled(rho=self.gmof_rho)(vertex_height)
        contact_confidence = torch.sum(pressure_weights, dim=1)
        # contact_mask = (vertex_height < self.contact_thresh).float()
        # num_contact_verts = torch.sum(contact_mask, dim=1)
        # contact_centroid_naive = torch.sum(vertices_hd * contact_mask[:, :, None], dim=1) / (torch.sum(contact_mask, dim=1) + eps)

        # project com, cop to ground plane (x-z plane)
        # weight loss by number of contact vertices to zero out if zero vertices in contact
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        contact_centroid_xz = torch.stack([cop[:, 0], torch.zeros_like(cop)[:, 0], cop[:, 2]], dim=1)
        # stability_loss = (contact_confidence * torch.norm(com_xz - contact_centroid_xz, dim=1)).sum(dim=-1)
        stability_loss = (torch.norm(com_xz - contact_centroid_xz, dim=1))
        return stability_loss
