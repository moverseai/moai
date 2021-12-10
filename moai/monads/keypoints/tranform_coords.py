import torch

import functools

__all__ = [
    "NormToCoords",
    "NormToNdc",
    "NdcToCameraCoords",
    "ScaleCoords",
    "UpscaleCoords_x2",
    "UpscaleCoords_x4",
    "DownscaleCoords_x2",
    "DownscaleCoords_x4",
    "CoordsToNorm",
]

#TODO: extract to generation/grid/conversions and refactor to support all cases with from/to arguments
class NormToCoords(torch.nn.Module):
    def __init__(self,
        mode: str="coord",
        flip: bool= False,
    ):
        super(NormToCoords,self).__init__()
        self.mode = mode
        self.flip = flip

    def forward(self, coords: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if self.flip:
            coords = coords * torch.Tensor([*img.shape[2:]]).flip(-1).to(coords).expand_as(coords)
        else:
            coords = coords * torch.Tensor([*img.shape[2:]]).to(coords).expand_as(coords)
        return coords

class NormToNdc(torch.nn.Module):
    def __init__(self,
        mode: str="coord",
        flip: bool= False,
    ):
        super(NormToNdc,self).__init__()
        self.mode = mode
        self.flip = flip

    def forward(self,coords: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if self.flip:
            ndc = torch.addcmul(torch.scalar_tensor(-1.0).to(coords), coords, torch.scalar_tensor(2.0).to(coords)).flip(-1), # from [0, 1] to [-1, 1]
        else:
            ndc = torch.addcmul(torch.scalar_tensor(-1.0).to(coords), coords, torch.scalar_tensor(2.0).to(coords)), # from [0, 1] to [-1, 1]
        return ndc[0]


class NdcToCameraCoords(torch.nn.Module):
    def __init__(
        self,
        fov:    float=64.69, # in degrees
        width:  int=320,
        height: int=240,
        flip:   bool= False,
        order:  str='xy' , # order of coords
    ):
        super(NdcToCameraCoords,self).__init__()
        self.flip = flip
        self.order = order
        self.tanfov = torch.tan((torch.deg2rad(torch.tensor(fov) / 2.0).float()))
        self.aspect_ratio = width / height
    
    def forward(self,coords: torch.Tensor) -> torch.Tensor:
        if self.order == "xy":
            camera_coords = torch.div(coords,torch.tensor([self.aspect_ratio / self.tanfov,(1 / self.tanfov)]).to(coords))
        else:
            camera_coords = torch.div(coords,torch.tensor([(1 / self.tanfov),self.aspect_ratio / self.tanfov]).to(coords))
        return camera_coords if not self.flip else camera_coords.flip(-1)

class ScaleCoords(torch.nn.Module):
    def __init__(self,
        scale:  float=0.5,
        flip:   bool= False,
    ):
        super(ScaleCoords,self).__init__()
        self.scale = scale
        self.flip = flip

    def forward(self,
        coords: torch.Tensor 
    ) -> torch.Tensor:
        coords = self.scale * coords
        if self.flip:
            return coords.flip(-1)
        else:
            return coords


class CoordsToNorm(torch.nn.Module):
    def __init__(self,
        flip: bool= False,
    ):
        super(CoordsToNorm,self).__init__()
        self.flip = flip

    def forward(self,
        coords: torch.Tensor,
        grid:   torch.Tensor
    ) -> torch.Tensor:
        dims = torch.tensor(grid.shape[2:], dtype=torch.float32, device=coords.device)
        if self.flip:
            dims = dims.flip(-1)
        return coords / dims.expand_as(coords)

UpscaleCoords_x2 = functools.partial(
    ScaleCoords, 
    scale=2
)

UpscaleCoords_x4 = functools.partial(
    ScaleCoords, 
    scale=4
)

DownscaleCoords_x2 = functools.partial(
    ScaleCoords, 
    scale=0.5
)

DownscaleCoords_x4 = functools.partial(
    ScaleCoords, 
    scale=0.25
)