import torch

class Projection(torch.nn.Module):

    __XYZ_AT__ = ['channel', 'row']
    __UV_AT__ = ['channel', 'row']
    __UV_TYPE__ = ['ndc', 'coord', 'norm']

    def __init__(self,
        xyz_at:         str='channel',  # ['channel', 'row]
        uv_at:          str='channel',  # ['channel', 'row]
        uv_type:        str='ndc',      # ['ndc', 'coord', 'norm']
    ):
        super(Projection, self).__init__()
        self.homogemize = self._homogemize_channel if xyz_at == 'channel'\
            else self._homogenize_row
        self.format_uv = self._format_uv_channel if uv_at == 'channel'\
            else self._format_uv_row
        self.convert_uv = self._convert_ndc if uv_type == 'ndc'\
            else (self._convert_norm if uv_type == 'norm'\
                else lambda x , h , w: x[:,:2,...]
            )

    def _homogemize_channel(self,
        points:             torch.Tensor,
        epsilon:            float=1e-8,
    ) -> torch.Tensor:
        return points / (points[:, -1:, ...] + epsilon)

    def _homogenize_row(self, 
        points:             torch.Tensor,
        epsilon:            float=1e-8,
    ) -> torch.Tensor:
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2] + epsilon
        x_h = x / z
        y_h = y / z
        ones = torch.ones_like(z)
        return torch.stack([x_h, y_h, ones], dim=1)

    def _convert_norm(self, 
        uvs:                torch.Tensor, 
        height:             int,
        width:              int,
    ) -> torch.Tensor:
        dims = torch.stack([
            torch.Tensor([[width]]), torch.Tensor([[height]])
        ], dim=1).to(uvs)
        return uvs[:, :2, ...] / dims

    def _convert_ndc(self, 
        uvs:                torch.Tensor, 
        height:             int,
        width:              int,
    ) -> torch.Tensor:
        one = self._convert_norm(uvs, height, width)
        return one * 2.0 - 1.0

    def _format_uv_channel(self, 
        uvs:                torch.Tensor, 
        height:             int,
        width:              int,
    ) -> torch.Tensor:
        b = uvs.shape[0]
        dims = [b, 2, height, width] if height > 1 else [b, 2, width]
        return uvs.reshape(*dims)

    def _format_uv_row(self, 
        uvs:                torch.Tensor,
        height:             int,
        width:              int,
    ) -> torch.Tensor:
        uv = uvs.transpose(-1, -2)
        b = uv.shape[0]
        dims = [b, height, width, 2] if height > 1 else [b, width, 2]
        return uv.reshape(*dims)

    def forward(self,
        intrinsics:         torch.Tensor,
        points:             torch.Tensor,
        grid:               torch.Tensor=None,
    ) -> torch.Tensor:
        b = points.shape[0]        
        homogeneous = self.homogemize(points)
        uvs = intrinsics @ homogeneous.view(b, 3, -1)
        grid = points if grid is None else grid
        w = grid.shape[-1]
        h = grid.shape[-2] 
        uvs = self.convert_uv(uvs, h, w)
        w_p = uvs.shape[-1]
        h_p = uvs.shape[-2] if len(uvs.shape) > 3 else 0
        return self.format_uv(uvs, h_p, w_p)

