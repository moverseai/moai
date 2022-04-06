from moai.monads.human.body.prior.human_body_prior import (
    VPoser_v1,
    _matrot2aa,
)

import torch
import typing

#NOTE: code from https://github.com/nghorbani/human_body_prior

__all__ = ["VPoser2"]

class VPoser1(VPoser_v1):
    def __init__(self,
        flatten_pose:       bool=True,
    ):
        super().__init__(
            num_neurons=512,
            latentD=32,
            data_shape=[1, 21, 3],
            use_cont_repr=True,
        )
        self.eval()
        self.flatten_pose = flatten_pose
    
    def forward(self,
        encode:         torch.Tensor=None,
        decode:         torch.Tensor=None,
        autoencode:     torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        out = { }        
        if autoencode is not None:
            # decoded = super(VPoser2, self).forward(autoencode)
            q_z = self.encode(autoencode)
            # q_z_sample = q_z.rsample()
            q_z_sample = q_z.mean
            dec = self.decode(q_z_sample)
            bs = autoencode.shape[0]
            decoded = {
                'pose_body': _matrot2aa(dec.view(-1, 3, 3)).view(bs, -1, 3),
                'pose_body_matrot': dec.view(bs, -1, 9),
            }
            decoded.update({
                'poZ_body_mean': q_z.mean,
                'poZ_body_std': q_z.scale,
                'q_z': q_z,
            })
            out['pose'] = decoded['pose_body']
            out['embedding'] = decoded['poZ_body_mean'] # decoded['q_z']
            if self.flatten_pose:
                out['pose'] = out['pose'].reshape(autoencode.shape[0], -1)
            return out
        if encode is not None:
            out['embedding'] = self.encode(encode)
        if decode is not None:
            dec = self.decode(decode)
            bs = decode.shape[0]            
            out['pose'] = _matrot2aa(dec.view(-1, 3, 3)).view(bs, -1, 3)
            if self.flatten_pose:
                out['pose'] = out['pose'].reshape(decode.shape[0], -1)
        return out

if __name__ == '__main__':
    vposer1 = VPoser1(True)
    emb = vposer1.forward(encode=torch.zeros(1, 21, 3))['embedding']
    aa = vposer1.forward(decode=torch.zeros(1, 32))['pose']
    ae = vposer1.forward(autoencode=torch.zeros(1, 21, 3))['pose']
    print(emb.mean.shape)
    print(aa.shape)
    print(ae.shape)
    vposer1 = VPoser1(False)
    emb = vposer1.forward(encode=torch.zeros(1, 21, 3))['embedding']
    aa = vposer1.forward(decode=torch.zeros(1, 32))['pose']
    ae = vposer1.forward(autoencode=torch.zeros(1, 21, 3))['pose']
    print(emb.mean.shape)
    print(aa.shape)
    print(ae.shape)

    ckpt = torch.load(r"E:/Data\SMPL/vposer_v1_0/snapshots/TR00_E096.pt")
    vposer1 = VPoser1(True)
    vposer1.load_state_dict(ckpt)
    emb = vposer1.forward(encode=torch.zeros(1, 21, 3))['embedding']
    aa = vposer1.forward(decode=torch.zeros(1, 32))['pose']
    ae = vposer1.forward(autoencode=torch.zeros(1, 21, 3))['pose']
    print(emb.mean.shape)
    print(aa.shape)
    print(ae.shape)
    vposer1 = VPoser1(False)
    emb = vposer1.forward(encode=torch.zeros(1, 21, 3))['embedding']
    aa = vposer1.forward(decode=torch.zeros(1, 32))['pose']
    ae = vposer1.forward(autoencode=torch.zeros(1, 21, 3))['pose']
    print(emb.mean.shape)
    print(aa.shape)
    print(ae.shape)