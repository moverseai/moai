import torch
import typing

class Transformation(torch.nn.Module):
    __XYZ_AT__ = ['channel', 'row']

    def __init__(self,
        xyz_in_at:              str='channel', # ['channel', 'row]
        xyz_out_at:             str='channel', # ['channel', 'row]
        transpose:              bool=False,
    ):
        super(Transformation, self).__init__()
        self.transpose = transpose
        self.convert_points_in = self._convert_points_row_in if xyz_in_at == 'row'\
            else self._convert_points_channel_in
        self.convert_points_out = self._convert_points_row_out if xyz_out_at == 'row'\
            else self._convert_points_channel_out
        self.get_width_height = self._get_width_height_row_in if xyz_in_at == 'row'\
            else self._get_width_height_channel_in

    def _convert_points_row_in(self,
        points:                 torch.Tensor,
    ):
        return points.permute(0, 2, 1) if len(points.shape) == 3\
            else points.permute(0, 3, 1, 2)
        
    def _convert_points_channel_in(self,
        points:                 torch.Tensor,
    ):
        return points

    def _get_width_height_row_in(self,
        shape:                 torch.Size,
    ):
        return shape[2], shape[1]

    def _get_width_height_channel_in(self,
        shape:                 torch.Size,
    ):
        return shape[3], shape[2]

    def _convert_points_channel_out(self,
        points:                 torch.Tensor,
        shape:                  torch.Size,
    ):        
        return points if len(shape) == 3\
            else points.reshape(
                points.shape[0], *self.get_width_height(shape), 3
            )
        
    def _convert_points_row_out(self,
        points:                 torch.Tensor,
        shape:                  torch.Size,
    ):
        pts = points.transpose(2, 1)
        return pts if len(shape) == 3 else points.reshape(
            points.shape[0], 3, *self.get_width_height(shape)
        )

    def forward(self, 
        points:                 torch.Tensor,      # [B, 3, H, W] or [B, H, W, 3] or [B, 3, V] or [B, V, 3] 
        transform:              typing.Optional[torch.Tensor] = None, # [B, 4, 4]
        rotation:               typing.Optional[torch.Tensor] = None, # [B, 3, 3]
        translation:            typing.Optional[torch.Tensor] = None, # [B, 3]
    )-> torch.Tensor:
        b = points.shape[0]
        if transform is not None:
            xform = torch.transpose(transform, -1, -2) if self.transpose else transform
            rot = xform[:, :3, :3]
            #trans = transform[:, 3, :3].reshape(b, 3, 1) #TODO: What if the trans vector is in the last row?
            trans = transform[:, :3, 3].reshape(transform.shape[0], 3, 1)
        else:
            rot = torch.transpose(rotation, 2, 1) if self.transpose else rotation
            trans = translation.reshape(translation.shape[0], 3, 1)
        pts3d = self.convert_points_in(points)
        xformed = rot @ pts3d.view(b, 3, -1) + trans
        return self.convert_points_out(xformed, points.size())
        
