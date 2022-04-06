from moai.monads.human.body.prior.human_body_prior import VPoser_v2
from collections import namedtuple

import torch
import typing

#NOTE: code from https://github.com/nghorbani/human_body_prior

__all__ = ["VPoser2"]

VPoser2InnerParams = namedtuple('VPoser2InnerParams', ['num_neurons', 'latentD'])
VPoser2Params = namedtuple('VPoser2Params', ['model_params'])

class VPoser2(VPoser_v2):
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
            # decoded = super(VPoser2, self).forward(autoencode)
            q_z = super(VPoser2, self).encode(autoencode)
            # q_z_sample = q_z.rsample()
            q_z_sample = q_z.mean
            decoded = super(VPoser2, self).decode(q_z_sample)
            decoded.update({
                'poZ_body_mean': q_z.mean,
                'poZ_body_std': q_z.scale,
                'q_z': q_z}
            )
            out['pose'] = decoded['pose_body']
            out['embedding'] = decoded['poZ_body_mean'] # decoded['q_z']
            if self.flatten_pose:
                out['pose'] = out['pose'].reshape(autoencode.shape[0], -1)
            return out
        if encode is not None:
            out['embedding'] = self.encode(encode)
        if decode is not None:
            out['pose'] = self.decode(decode)['pose_body']
            if self.flatten_pose:
                out['pose'] = out['pose'].reshape(decode.shape[0], -1)
        return out

