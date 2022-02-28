from moai.utils.arguments import assert_positive
import torch
import typing
import logging

logger = logging.getLogger(__name__)

__all__ = ["WeakPerspective"]

class WeakPerspective(torch.nn.Module): #NOTE: fixed focal/principal, optimized rot/trans
    def __init__(self,
        focal_length:       typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:    typing.Union[float, typing.Tuple[float, float]]=0.5,
        width:              int=None,
        height:             int=None,
        rotation:           torch.Tensor=None,
        translation:        torch.Tensor=None,
        persistent:         bool=False,
    ):
        super(WeakPerspective, self).__init__()
        mat = torch.zeros([1, 2, 2])
        mat[:, 0, 0] = focal_length if isinstance(focal_length, float) else focal_length[0]
        mat[:, 1, 1] = focal_length if isinstance(focal_length, float) else focal_length[1]
        self.register_buffer("mat", mat, persistent=persistent)
        center = torch.zeros([1, 2])
        if principal_point is None:
            assert_positive(logger, "width", width)
            assert_positive(logger, "height", height)
            center[:, 0] = width // 2
            center[:, 1] =  height // 2
        else:
            center[:, 0] = torch.scalar_tensor(
                principal_point if isinstance(principal_point, float) else principal_point[0]
            )
            center[:, 1] = torch.scalar_tensor(
                principal_point if isinstance(principal_point, float) else principal_point[1]
            )
        self.register_buffer('principal_point', center, persistent=persistent)                                                         
        if rotation is None:
             rotation = torch.eye(3).unsqueeze(dim=0)
        rotation = torch.nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)
        if translation is None:
            translation = torch.zeros([1, 3])
        translation = torch.nn.Parameter(translation, requires_grad=True)
        self.register_parameter('translation', translation)

    def forward(self, 
        points:             torch.Tensor,
        image:              torch.Tensor=None,
        rotation:           torch.Tensor=None,
        translation:        torch.Tensor=None,
        #TODO: update with focal/principal inputs as well        
    ) -> torch.Tensor:
        if image is not None:
            h, w = image.shape[-2:]
            with torch.no_grad():
                self.principal_point[:, 0] = w // 2
                self.principal_point[:, 1] = h // 2
        R = rotation if rotation is not None else self.rotation.expand(points.shape[0], 3, 3)
        t = translation if translation is not None else self.translation.expand(points.shape[0], 3)
        camera_transform = torch.cat([
            torch.nn.functional.pad(R, [0, 0, 0, 1]),
            torch.nn.functional.pad(t.unsqueeze(dim=-1), [0, 0, 0, 1], value=1)
        ], dim=2)
        z = torch.ones_like(points[..., :1])
        homogeneous_points = torch.cat([points, z], dim=-1)
        projected_points = torch.einsum('bki,bji->bjk',
            [camera_transform, homogeneous_points]
        )
        img_points = torch.div(projected_points[:, :, :2],
            projected_points[:, :, 2].unsqueeze(dim=-1)
        )
        img_points = torch.einsum('bki,bji->bjk', [self.mat, img_points]) \
            + self.principal_point.unsqueeze(dim=1) #TODO: add principal in mat
        return img_points