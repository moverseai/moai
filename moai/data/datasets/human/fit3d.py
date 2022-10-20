from moai.utils.arguments import assert_path

import functools
import toolz
import numpy as np
import glob
import torch
import os
import typing
import logging
import tqdm
import smplx
import orjson

log = logging.getLogger(__name__)

__all__ = ['Fit3D']

# try:
#     import orjson
#     @lru_cache(maxsize=None)
#     #@functools.lru_cache(maxsize=None)
#     def _load_json(filename: str) -> dict:
#         with open(filename, 'rb') as f:
#             data = orjson.loads(f.read())
#         return data
# except:
#     log.warning("Could not load the `orjson` package which will improve loading speeds, please consider `pip install orjson`.")
#     import json
#     @functools.lru_cache(maxsize=None)
#     def _load_json(filename: str) -> dict:
#         with open(filename, 'rb') as f:
#             data = json.load(f)
#         return data

class Fit3D(torch.utils.data.Dataset):
    def __init__(self,
        data_root:          str,
        smplx_root:         str,
        subjects:           typing.Union[typing.List[str], str]='**',
        actions:            typing.Union[typing.List[str], str]='**',
        downsample_factor:  int=1,
        device:             typing.Union[int, typing.Sequence[int]]=[-1],
        reconstruct:        bool=False,
    ) -> None:
        super().__init__()
        assert_path(log, 'Fit3D data root path', data_root)
        assert_path(log, 'SMPLX data path', smplx_root)
        is_all_subjects = isinstance(subjects, str) and ('all' == subjects or '**' == subjects)
        subjects = os.listdir(data_root) if is_all_subjects else subjects
        subjects = [subjects] if isinstance(subjects, str) else subjects
        is_all_actions = isinstance(actions, str) and ('all' == actions or '**' == actions)
        actions = list(map(lambda fn: os.path.splitext(fn)[0], 
            toolz.unique(map(
                os.path.basename, 
                glob.glob(os.path.join(data_root, '**', 'smplx', '*.json'))
            ))
        )) if is_all_actions else actions
        actions = [actions] if isinstance(actions, str) else actions
        self.device = f"cuda:{device[0]}" if isinstance(device, typing.Sequence) and device[0] >= 0 else 'cpu'
        self.data = {
            'transl': torch.empty((0, 3)),
            'global_orient': torch.empty((0, 9)),
            'body_pose': torch.empty((0, 21 * 9)),
            'betas': torch.empty((0, 10)),
            'left_hand_pose': torch.empty((0, 15 * 9)),
            'right_hand_pose': torch.empty((0, 15 * 9)),
            'jaw_pose': torch.empty((0, 9)),
            'leye_pose': torch.empty((0, 9)),
            'reye_pose': torch.empty((0, 9)),
            'expression': torch.empty((0, 10)),
         }
        self._load_json = functools.lru_cache()(self.__load_json)
        for subj in tqdm.tqdm(subjects, desc='Loading Fit3D subjects'):                        
            for action_fn in tqdm.tqdm(
                glob.glob(os.path.join(data_root, subj, 'smplx', "*.json")),
                desc='Loading Fit3D subject actions'
            ):
                if os.path.splitext(os.path.basename(action_fn))[0] not in actions:
                    continue
                data = self._load_json(action_fn)
                for k, v in data.items():
                    self.data[k] = torch.cat([
                        self.data[k], 
                        torch.from_numpy(np.array(v, dtype=np.float32).squeeze())[::downsample_factor, ...].flatten(1)
                    ])                                        
        self.reconstruct = reconstruct
        if self.reconstruct:
            self.body = smplx.SMPLXLayer(
                model_path=os.path.join(smplx_root, 'smplx'), model_type='smplx',
                num_expression_coeffs=10, batch_size=1, age='adult', 
                use_pca=False, flat_hand_mean=False,
                num_betas=10, gender='neutral'
            ).to(self.device)
            self.body.requires_grad_(False)
        log.info(f"Loaded {len(self)} Fit3D samples.")

    def __load_json(self,filename: str) -> dict:
        with open(filename, 'rb') as f:
            data = orjson.loads(f.read())
        return data

    def __len__(self) -> int:
        return len(self.data['betas'])

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = toolz.itemmap(lambda d: (d[0], d[1][index]), self.data)
        out = {
            'smplx': {
                'gender': 0,
                'scale': torch.scalar_tensor(1.0), 
                'params': {                
                    'betas': item['betas'],
                    'body_pose': item['body_pose'],
                    'transl': item['transl'],
                    'global_orient': item['global_orient'],
                    'left_hand_pose': item['left_hand_pose'],
                    'right_hand_pose': item['right_hand_pose'],
                    'jaw_pose': item['jaw_pose'],
                    'left_eye_pose': item['leye_pose'],
                    'right_eye_pose': item['reye_pose'],
                }
            },
            '__moai__': {
                'dataset': type(self).__name__,
            }
        }
        if self.reconstruct:
            with torch.no_grad():
                body = self.body.forward(
                    global_orient=out['smplx']['params']['global_orient'][np.newaxis, ...].to(self.device),
                    betas=out['smplx']['params']['betas'][np.newaxis, ...].to(self.device),
                    body_pose=out['smplx']['params']['body_pose'][np.newaxis, ...].to(self.device),
                    left_hand_pose=out['smplx']['params']['left_hand_pose'][np.newaxis, ...].to(self.device),
                    right_hand_pose=out['smplx']['params']['right_hand_pose'][np.newaxis, ...].to(self.device),
                    jaw_pose=out['smplx']['params']['jaw_pose'][np.newaxis, ...].to(self.device),
                    leye_pose=out['smplx']['params']['left_eye_pose'][np.newaxis, ...].to(self.device),
                    reye_pose=out['smplx']['params']['right_eye_pose'][np.newaxis, ...].to(self.device),
                    transl=out['smplx']['params']['transl'][np.newaxis, ...].to(self.device),
                    pose2rot=False,
                )
                out['smplx'].update({
                    'mesh': {
                        'vertices': body.vertices[0].cpu(),                
                        'faces': self.body.faces_tensor.cpu(),
                        },                
                    'joints': body.joints[0, ...].cpu(),

                })
        return out