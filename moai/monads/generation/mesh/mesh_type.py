import torch

#NOTE: modified code from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/rep/Mesh.py

__all__ = ["Mesh", "TriangleMesh"]

class Mesh():
    """ Abstract class to represent 3D polygon meshes. """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor,
                 uvs: torch.Tensor, face_textures: torch.Tensor,
                 textures: torch.Tensor, edges: torch.Tensor, edge2key: dict, vv: torch.Tensor,
                 vv_count: torch.Tensor, vf: torch.Tensor, vf_count: torch.Tensor,
                 ve: torch.Tensor, ve_count: torch.Tensor, ff: torch.Tensor,
                 ff_count: torch.Tensor, ef: torch.Tensor, ef_count: torch.Tensor,
                 ee: torch.Tensor, ee_count: torch.Tensor):\

        # Vertices of the mesh
        self.vertices = vertices
        # Faces of the mesh
        self.faces = faces
        # uv coordinates of each vertex
        self.uvs = uvs
        # uv indecies for each face
        self.face_textures = face_textures
        # texture for each face
        self.textures = textures
        # Edges of the mesh
        self.edges = edges
        # Dictionary that maps an edge (tuple) to an edge idx
        self.edge2key = edge2key
        # Vertex-Vertex neighborhood tensor (for each vertex, contains
        # indices of the vertices neighboring it)
        self.vv = vv
        # Number of vertices neighbouring each vertex
        self.vv_count = vv_count
        # Vertex-Face neighborhood tensor
        self.vf = vf
        # Number of faces neighbouring each vertex
        self.vf_count = vf_count
        # Vertex-Edge neighborhood tensor
        self.ve = ve
        # Number of edges neighboring each vertex
        self.ve_count = ve_count
        # Face-Face neighborhood tensor
        self.ff = ff
        # Number of faces neighbouring each face
        self.ff_count = ff_count
        # Edge-Face neighbourhood tensor
        self.ef = ef
        # Number of edges neighbouring each face
        self.ef_count = ef_count
        # Edge-Edge neighbourhood tensor
        self.ee = ee
        # Number of edges neighbouring each edge
        self.ee_count = ee_count
        # adjacency matrix for verts
        self.adj = None
    
    @classmethod
    def from_obj(self, filename: str, with_vt: bool = False,
                 enable_adjacency: bool = False, texture_res=4):
        r"""Loads object in .obj wavefront format.
        Args:
            filename (str) : location of file.
            with_vt (bool): objects loaded with textures specified by vertex
                textures.
            enable_adjacency (bool): adjacency information is computed.
            texture_res (int): resolution of loaded face colors.
        Note: the with_vt parameter requires cuda.
        Example:
            >>> mesh = Mesh.from_obj('model.obj')
            >>> mesh.vertices.shape
            torch.Size([482, 3])
            >>> mesh.faces.shape
            torch.Size([960, 3])
        """
        # run through obj file and extract obj info
        vertices = []
        faces = []
        face_textures = []
        uvs = []
        with open(filename, 'r') as mesh:
            for line in mesh:
                data = line.split()
                if len(data) == 0:
                    continue
                if data[0] == 'v':
                    vertices.append(data[1:])
                elif data[0] == 'vt':
                    uvs.append(data[1:3])
                elif data[0] == 'f':
                    if '//' in data[1]:
                        data = [da.split('//') for da in data]
                        faces.append([int(d[0]) for d in data[1:]])
                        face_textures.append([int(d[1]) for d in data[1:]])
                    elif '/' in data[1]:
                        data = [da.split('/') for da in data]
                        faces.append([int(d[0]) for d in data[1:]])
                        face_textures.append([int(d[1]) for d in data[1:]])
                    else:
                        faces.append([int(d) for d in data[1:]])
                        continue
        vertices = torch.FloatTensor([float(el) for sublist in vertices for el in sublist]).view(-1, 3)
        faces = torch.LongTensor(faces) - 1

        # compute texture info
        textures = None
        if with_vt:
            with open(filename, 'r') as f:
                textures = None
                for line in f:
                    if line.startswith('mtllib'):
                        filename_mtl = os.path.join(
                            os.path.dirname(filename), line.split()[1])
                        textures = self.load_textures(
                            filename, filename_mtl, texture_res)

                f.close()

        if len(uvs) > 0:
            uvs = torch.FloatTensor([float(el) for sublist in uvs for el in sublist]).view(-1, 2)
        else:
            uvs = None
        if len(face_textures) > 0:
            face_textures = torch.LongTensor(face_textures) - 1
        else:
            face_textures = None

        if enable_adjacency:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, ff_count, \
                ee, ee_count, ef, ef_count = self.compute_adjacency_info(
                    vertices, faces)
        else:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, \
                ff_count, ee, ee_count, ef, ef_count = None, None, None, \
                None, None, None, None, None, None, None, None, None, None, \
                None

        output = self(vertices, faces, uvs, face_textures, textures, edges,
                    edge2key, vv, vv_count, vf, vf_count, ve, ve_count, ff, ff_count,
                    ef, ef_count, ee, ee_count)
        
        return output


class TriangleMesh(Mesh):
    """ Abstract class to represent 3D Trianlge meshes. """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor,
                 uvs: torch.Tensor, face_textures: torch.Tensor,
                 textures: torch.Tensor, edges: torch.Tensor, edge2key: dict,
                 vv: torch.Tensor, vv_count: torch.Tensor,
                 vf: torch.Tensor, vf_count: torch.Tensor,
                 ve: torch.Tensor, ve_count: torch.Tensor,
                 ff: torch.Tensor, ff_count: torch.Tensor,
                 ef: torch.Tensor, ef_count: torch.Tensor,
                 ee: torch.Tensor, ee_count: torch.Tensor):

        # Vertices of the mesh
        self.vertices = vertices
        # Faces of the mesh
        self.faces = faces
        # uv coordinates of each vertex
        self.uvs = uvs
        # uv indecies for each face
        self.face_textures = face_textures
        # texture for each face
        self.textures = textures
        # Edges of the mesh
        self.edges = edges
        # Dictionary that maps an edge (tuple) to an edge idx
        self.edge2key = edge2key
        # Vertex-Vertex neighborhood tensor (for each vertex, contains
        # indices of the vertices neighboring it)
        self.vv = vv
        # Number of vertices neighbouring each vertex
        self.vv_count = vv_count
        # Vertex-Face neighborhood tensor
        self.vf = vf
        # Number of faces neighbouring each vertex
        self.vf_count = vf_count
        # Vertex-Edge neighborhood tensor
        self.ve = ve
        # Number of edges neighboring each vertex
        self.ve_count = ve_count
        # Face-Face neighborhood tensor
        self.ff = ff
        # Number of faces neighbouring each face
        self.ff_count = ff_count
        # Edge-Face neighbourhood tensor
        self.ef = ef
        # Number of edges neighbouring each face
        self.ef_count = ef_count
        # Edge-Edge neighbourhood tensor
        self.ee = ee
        # Number of edges neighbouring each edge
        self.ee_count = ee_count
        # adjacency matrix for verts
        self.adj = None


