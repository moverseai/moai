import torch
import typing

from collections import namedtuple
from human_body_prior.models.vposer_model import VPoser #TODO: try/except and error msg

#NOTE: code from https://github.com/nghorbani/human_body_prior

__all__ = ["VPoser2"]

VPoser2InnerParams = namedtuple('VPoser2InnerParams', ['num_neurons', 'latentD'])
VPoser2Params = namedtuple('VPoser2Params', ['model_params'])

class VPoser2(VPoser):
    def __init__(self,
        flatten_pose:       bool=True,
    ):
        super(VPoser2, self).__init__(
            VPoser2Params(VPoser2InnerParams(512, 32))
        )
        self.flatten_pose = flatten_pose
        self.eval()

    def forward(self,
        encode:         torch.Tensor=None,
        decode:         torch.Tensor=None,
        autoencode:     torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        out = { }        
        if autoencode is not None:
            out['embedding'] = self.encode(encode)
            out['pose'] = self.decode(decode)
            return out
        if encode is not None:
            out['embedding'] = self.encode(encode)
        if decode is not None:
            out['pose'] = self.decode(decode)['pose_body']
            if self.flatten_pose:
                out['pose'] = out['pose'].reshape(decode.shape[0], -1)
        return out

