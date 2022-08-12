from moai.utils.arguments import assert_positive
import torch
import typing
import logging
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Camera"]

class Camera(torch.nn.Module): #NOTE: fixed focal/principal, optimized rot/trans
    def __init__(self,
        focal_length:       typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:    typing.Union[float, typing.Tuple[float, float]]=0.5,
        width:              int=None,
        height:             int=None,
        rotation:           torch.Tensor=None,
        translation:        torch.Tensor=None,
        persistent:         bool=False,
    ):
        super(Camera, self).__init__()
        self.focal_length = (focal_length, focal_length) if isinstance(focal_length, float) else focal_length
        self.principal_point = (principal_point, principal_point) if isinstance(principal_point, float) else principal_point
        self.resolution = (width, height)
        if principal_point is None:
            assert_positive(logger, "width", width)
            assert_positive(logger, "height", height)
            self.principal_point = (float(width // 2), float(height // 2))
            self.has_nominal_principal = True
        else:
            self.has_nominal_principal = False
        sx = 0.0
        x0, y0 = (0.0, 0.0)
        n, f = (0.1, 50.0)
        w, h = width, height
        fx, fy = self.focal_length
        px, py = self.principal_point
        mat = torch.tensor([[
            [2 * fx / w,                            0,                                  0,                                  0],
            [-2.0 * sx / w,                    -2.0 * fy / h,                             0,                                  0],
            [(w - 2.0 * px + 2 * x0) / w,   (h - 2.0 * py + 2.0 * y0) / h,        -(f + n) / (f - n),               -1.0],
            [0,                                         0,                     -(2.0 * f * n) / (f - n),              0]
        ]])
        self.register_buffer("mat", mat, persistent=persistent)
        if rotation is None:
             rotation = torch.eye(3)[np.newaxis, ...].float()
        rotation = torch.nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)
        if translation is None:
            translation = torch.zeros([1, 3]).float()
        translation = torch.nn.Parameter(translation, requires_grad=True)
        self.register_parameter('translation', translation)

    def forward(self, 
        points:             torch.Tensor,
        image:              torch.Tensor=None,
        nominal_image:      torch.Tensor=None,
        aspect_image:       torch.Tensor=None,
        rotation:           torch.Tensor=None,
        translation:        torch.Tensor=None,
        intrinsics:         torch.Tensor=None,
        #TODO: update with focal/principal inputs as well        
    ) -> torch.Tensor:
        fx, fy = self.focal_length
        px, py = self.principal_point
        if image is not None:            
            w, h = image.shape[-1], image.shape[-2]
            sx = 0.0
            x0, y0 = (0.0, 0.0)
            n, f = (0.1, 50.0)
            px, py = w / 2.0, h / 2.0
            proj = torch.tensor([[
                [2 * fx / w,                               0,                                  0,                                  0],
                [-2.0 * sx / w,                    -2.0 * fy / h,                             0,                                  0],
                [(w - 2.0 * px + 2 * x0) / w,   (h - 2.0 * py + 2.0 * y0) / h,        -(f + n) / (f - n),               -1.0],
                [0,                                         0,                     -(2.0 * f * n) / (f - n),              0]
            ]]).to(points)
        elif nominal_image is not None:
            w, h = self.resolution
            ow, oh = nominal_image.shape[-1], nominal_image.shape[-2]
            fx, fy = fx * (w / ow), fy * (h / oh)
            sx = 0.0
            x0, y0 = (0.0, 0.0)
            n, f = (0.1, 50.0)
            if not self.has_nominal_principal:
                px, py = (px / ow) * w, (py / oh) * h
            else:
                px, py = 0.5 * w, 0.5 * h
            proj = torch.tensor([[
                [2 * fx / w,                               0,                                  0,                                  0],
                [-2.0 * sx / w,                    -2.0 * fy / h,                             0,                                  0],
                [(w - 2.0 * px + 2 * x0) / w,   (h - 2.0 * py + 2.0 * y0) / h,        -(f + n) / (f - n),               -1.0],
                [0,                                         0,                     -(2.0 * f * n) / (f - n),              0]
            ]]).to(points)
        elif aspect_image is not None:
            w, h = self.resolution
            ow, oh = aspect_image.shape[-1], aspect_image.shape[-2]
            if oh > ow:
                fx, fy = fx * (w / ow) / ((h / w) / (oh / ow)), fy * (h / oh)
                w = ow * (h / oh)
            else:                
                fx, fy = fx * (w / ow), fy * (h / oh) / ((h / w) / (oh / ow))
                h = oh * (w / ow)
            sx = 0.0
            x0, y0 = (0.0, 0.0)
            n, f = (0.1, 50.0)
            if not self.has_nominal_principal:
                px, py = (px / ow) * w, (py / oh) * h
            else:
                px, py = 0.5 * w, 0.5 * h
            proj = torch.tensor([[
                [2 * fx / w,                               0,                                  0,                                  0],
                [-2.0 * sx / w,                    -2.0 * fy / h,                             0,                                  0],
                [(w - 2.0 * px + 2 * x0) / w,   (h - 2.0 * py + 2.0 * y0) / h,        -(f + n) / (f - n),               -1.0],
                [0,                                         0,                     -(2.0 * f * n) / (f - n),              0]
            ]]).to(points)
        elif intrinsics is not None:
            Ks = []
            for K in intrinsics:
                w, h = self.resolution
                sx = 0.0
                x0, y0 = (0.0, 0.0)
                n, f = (0.1, 50.0)
                fx, fy = K[0, 0], K[1, 1]
                px, py = K[0, 2], K[1, 2]
                Ks.append(torch.tensor([
                    [2 * fx / w,                               0,                                  0,                                  0],
                    [-2.0 * sx / w,                    -2.0 * fy / h,                             0,                                  0],
                    [(w - 2.0 * px + 2 * x0) / w,   (h - 2.0 * py + 2.0 * y0) / h,        -(f + n) / (f - n),               -1.0],
                    [0,                                         0,                     -(2.0 * f * n) / (f - n),              0]
                ]).to(points))
            proj = torch.stack(Ks)
        else:
            proj = self.mat
            w, h = self.resolution
        b = points.shape[0]
        Rt = torch.zeros(b, 4, 4).to(points.device)
        t = translation if translation is not None else self.translation.expand(b, 3)
        R = rotation if rotation is not None else self.rotation.expand(b, 3, 3)
        Rt[:, :3, :3] = R
        Rt[:, 3, 0] = -1.0 * t[:, 0]
        Rt[:, 3, 1] = 1.0 * t[:, 1]
        Rt[:, 3, 2] = 1.0 * t[:, 2]
        Rt[:, 3, 3] = 1.0
        inv_Rt = torch.inverse(Rt)
        v = torch.nn.functional.pad(
            points, pad=(0,1), mode='constant', value=1.0
        ) if points.shape[-1] == 3 else points # [B, V, 4]
        xf = torch.einsum('bvi,bij->bvj', v, inv_Rt)
        ndc = torch.einsum('bvi,bij->bvj', xf, proj)
        return ndc