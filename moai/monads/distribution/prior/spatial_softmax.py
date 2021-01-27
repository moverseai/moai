from moai.monads.utils import flatten_spatial_dims

import torch

__all__ = ["SpatialSoftmax"]

#NOTE: see "FlowCap: 2D Human Pose from Optical Flow" for sharpening
#NOTE: see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

class SpatialSoftmax(torch.nn.Module):
    def __init__(self,
        temperature:    float=1.0,  # smoothen out the output by premultiplying input
        alpha:          float=1.0,  # sharpen output
        normalize:      bool=False, # normalize output
    ):
        super(SpatialSoftmax, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.normalize = normalize
        #TODO: add inplace version / flag

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:        
        reduced = flatten_spatial_dims(tensor)
        if self.temp != 1.0:
            reduced = reduced * self.temp
        if self.alpha != 1.0:
            reduced = reduced ** self.alpha
        if self.normalize:
            reduced = reduced / reduced.flatten(2).sum(-1)
        softmaxed = torch.nn.functional.softmax(reduced, dim=-1)
        return softmaxed.view_as(tensor)