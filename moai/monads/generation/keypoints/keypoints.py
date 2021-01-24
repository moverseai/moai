import torch
import typing

__all__ = ["Keypoints"]

class Keypoints(torch.nn.Module):
    def __init__(self,
        keypoints: typing.Union[str,typing.List[typing.List[float]]],
    ):
        super(Keypoints,self).__init__()
        #TODO: add numpy loading
        if isinstance(keypoints,str):
            #TODO:load keypoints from file
            pass
        self.register_buffer("keypoints", torch.tensor(keypoints).float())

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.keypoints.expand(b,*self.keypoints.shape)