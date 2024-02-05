from moai.utils.arguments import assert_path
from collections import OrderedDict

import functools
import toolz
import numpy as np
import glob
import torch
import os
import typing
import logging
import bisect
import smplx

log = logging.getLogger(__name__)

__all__ = ['AMASS']

class AMASS(torch.utils.data.Dataset):
    def __init__(self,
        data_root:          str,
        smplx_root:         str='',
        device:             list=[-1],
        parts:              typing.Union[typing.List[str], str]='**',
        downsample_factor:  int=1,
        reconstruct:        bool=False,
        model_type:         str='smplx',
    ) -> None:
        super().__init__()
        assert_path(log, 'AMASS data root path', data_root)
        assert_path(log, 'SMPLX data path', smplx_root)
        file_type = '**_poses.npz' if model_type=='smpl' else '**_stageii.npz'
        gendered_shape_fn = '**shape.npz' if model_type=='smpl' else '**_stagei.npz'
        is_all_parts = isinstance(parts, str) and ('all' == parts or '**' == parts)
        parts = os.listdir(data_root) if is_all_parts else parts
        parts = [parts] if isinstance(parts, str) else parts
        self.items = OrderedDict()
        self.sampling = downsample_factor
        frame_counter = 0
        self.subjects = { }
        self.device = device[0] if device[0] >= 0 else 'cpu'
        for part in parts:
            for subject in filter(os.path.isdir, glob.glob(os.path.join(data_root, part, '**'))):
                subject_name = os.path.basename(subject)
                if model_type == 'smplx': 
                    gendered_shape_fn = toolz.first(glob.glob(
                        os.path.join(data_root, part, subject_name, gendered_shape_fn)
                    ))
                    gender = os.path.basename(gendered_shape_fn).split('_')[0]
                for action_fn in glob.glob(os.path.join(data_root, part, subject, file_type)):
                    data = np.load(action_fn, allow_pickle=False)
                    shape = torch.from_numpy(data['betas']).float() if model_type == 'smpl' else torch.from_numpy(
                                                                                        np.load(
                                                                                            gendered_shape_fn,
                                                                                            allow_pickle=False
                                                                                        )['betas']
                                                                                    ).float().clone()
                    #frame_counter += int(data['trans'].shape[0] / downsample_factor)
                    frame_counter += max(1, int(data['trans'].shape[0] / downsample_factor))
                    self.items[frame_counter] = { 
                        'data': data, # action_fn, # data, 
                        'gender': str(data['gender']) if model_type == 'smpl' else gender,
                        'subject': f"{part}_{subject_name}",
                    }
                    del data
                self.subjects[f"{part}_{subject_name}"] = shape
        self.frame_keys = list(self.items.keys())
        self.reconstruct = reconstruct
        if self.reconstruct:
            with torch.no_grad():
                if model_type == 'smplx': 
                    smplx_create = functools.partial(smplx.create, 
                        model_path=smplx_root, model_type=model_type,
                        num_expression_coeffs=10, create_expression=True,
                        create_jaw_pose=False, create_leye_pose=False,
                        create_reye_pose=False, use_face_contour=False,
                        batch_size=1, age='adult', create_left_hand_pose=False,
                        create_right_hand_pose=False, use_pca=False,            
                        flat_hand_mean=False, create_betas=False,
                        num_betas=16, create_global_orient=False,
                        create_body_pose=False, create_transl=False,
                    )
                else:
                    smplx_create = functools.partial(smplx.create, 
                        model_path=smplx_root, model_type=model_type,
                        batch_size=1, age='adult', create_betas=False,
                        num_betas=10, create_global_orient=False,
                        create_body_pose=False, create_transl=False,
                    )
                self.bodies = {
                    'female': smplx_create(gender='female').to(self.device),
                    'male': smplx_create(gender='male').to(self.device),
                    'neutral': smplx_create(gender='neutral').to(self.device),
                    'Trial': smplx_create(gender='male').to(self.device)
                }
                del smplx_create
            for v in self.bodies.values():
                v.requires_grad_(False)
        log.info(f"Loaded {len(self)} AMASS samples.")
        self.model_type = model_type

    def __len__(self) -> int:
        return toolz.last(self.items)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        ind = bisect.bisect_left(self.frame_keys, index)
        ind += (self.frame_keys[ind] == index)
        frame = index - (self.frame_keys[ind-1] if ind > 0 else 0)
        frame *= self.sampling
        item = self.items[self.frame_keys[ind]]
        gender = item['gender']
        betas = self.subjects[item['subject']]
        data = item['data']
        #frame = min(frame,(len(data['pose_body'])-1))
        # data = np.load(item['data'], allow_pickle=False)
        if self.model_type == 'smplx':
            lhand, rhand = torch.from_numpy(data['pose_hand']).split(45, dim=1)
            leye, reye = torch.from_numpy(data['pose_eye']).split(3, dim=1)
            out = {
                'smplx': {
                    'gender': 1 if gender == 'male' else 2,
                    'scale': torch.scalar_tensor(1.0), 
                    'params': {                
                        'betas': betas,
                        'body_pose': torch.from_numpy(data['pose_body'][frame]).float().clone(), # 63
                        'transl': torch.from_numpy(data['trans'][frame]).float().clone(), # 3
                        'global_orient': torch.from_numpy(data['root_orient'][frame]).float().clone(), # 3
                        'left_hand_pose': lhand[frame].float().clone(), # 45
                        'right_hand_pose': rhand[frame].float().clone(), # 45
                        'jaw_pose': torch.from_numpy(data['pose_jaw'][frame]).float().clone(), # 3
                        'left_eye_pose': leye[frame].float().clone(), # 3
                        'right_eye_pose': reye[frame].float().clone(), # 3
                    },
                },
                '__moai__': {
                    'dataset': type(self).__name__,
                }
            }
        else:
            lhand, rhand = torch.from_numpy(data['poses'][:,66:]).split(45, dim=1)
            out = {
                'smpl': {
                    'gender': 1 if gender == 'male' else 2,
                    'scale': torch.scalar_tensor(1.0), 
                    'params': {                
                        'betas': betas[:10],
                        'body_pose': torch.from_numpy(data['poses'][frame,3:72]).float().clone(), # 63 (21x3) plus 2 x 3 for hands
                        'transl': torch.from_numpy(data['trans'][frame]).float().clone(), # 3
                        'global_orient': torch.from_numpy(data['poses'][frame,:3]).float().clone(), # 3
                        'left_hand_pose': lhand[frame].float().clone(), # 45
                        'right_hand_pose': rhand[frame].float().clone(), # 45
                    },
                },
                '__moai__': {
                    'dataset': type(self).__name__,
                }
            }
        if self.reconstruct:
            if self.model_type == 'smplx':
                body = self.bodies[gender].forward(
                    global_orient=out['smplx']['params']['global_orient'][np.newaxis, ...].to(self.device),
                    betas=out['smplx']['params']['betas'][np.newaxis, ...].to(self.device),
                    body_pose=out['smplx']['params']['body_pose'][np.newaxis, ...].to(self.device),
                    left_hand_pose=out['smplx']['params']['left_hand_pose'][np.newaxis, ...].to(self.device),
                    right_hand_pose=out['smplx']['params']['right_hand_pose'][np.newaxis, ...].to(self.device),
                    jaw_pose=out['smplx']['params']['jaw_pose'][np.newaxis, ...].to(self.device),
                    leye_pose=out['smplx']['params']['left_eye_pose'][np.newaxis, ...].to(self.device),
                    reye_pose=out['smplx']['params']['right_eye_pose'][np.newaxis, ...].to(self.device),
                    transl=out['smplx']['params']['transl'][np.newaxis, ...].to(self.device),
                )
            else:
                body = self.bodies[gender].forward(
                    global_orient=out['smpl']['params']['global_orient'][np.newaxis, ...].to(self.device),
                    betas=out['smpl']['params']['betas'][np.newaxis, ...].to(self.device),
                    body_pose=out['smpl']['params']['body_pose'][np.newaxis, ...].to(self.device),
                    left_hand_pose=out['smpl']['params']['left_hand_pose'][np.newaxis, ...].to(self.device),
                    right_hand_pose=out['smpl']['params']['right_hand_pose'][np.newaxis, ...].to(self.device),
                    transl=out['smpl']['params']['transl'][np.newaxis, ...].to(self.device),
                )
            out[self.model_type].update({
                'mesh': {
                    'vertices': body.vertices[0].cpu(),                
                    'faces': self.bodies[gender].faces_tensor.cpu(),
                    },
                'joints': body.joints[0].cpu(),
            })
        return out