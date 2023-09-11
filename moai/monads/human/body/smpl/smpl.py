from moai.monads.human.pose.openpose import JointMap

import toolz
import torch
import smplx #TODO: try/except and error msg
import functools
import typing
import numpy as np
from smplx.utils import Struct
from smplx import SMPLH as slh
from smplx.vertex_ids import vertex_ids



#NOTE: code from https://github.com/vchoutas/smplify-x

__all__ = ["SMPLX"]

__JOINT__MAPPERS__ = {
    'none':             None,
    'openpose_coco25':  functools.partial(JointMap, 
        model='smpl', format='coco25', with_hands=False,
        with_face=False, with_face_contour=False,
    ),
    'openpose_coco19':  functools.partial(JointMap,
        model='smpl', format='coco19', with_hands=False,
        with_face=False, with_face_contour=False,
    ),
}

class SMPL(smplx.SMPL):
    def __init__(self,
        model_path:             str,
        joints_format:          str='none',                
        gender:                 str='neutral',
        age:                    str='adult',
        num_betas:              int=10,
        use_global_orientation: bool=True, # root rotation
        use_translation:        bool=True, # body translation
        use_pose:               bool=True, # joint rotations, False when using VPoser
        use_betas:              bool=True, # shape
        batch_size:             int=1,
    ):
        mapper = __JOINT__MAPPERS__.get(joints_format, None)
        super().__init__(
            model_path=model_path,
            joint_mapper=None if joints_format is None else\
                mapper() if mapper is not None else None,
            create_global_orient=use_global_orientation,
            create_body_pose=use_pose,
            create_betas=use_betas,
            create_transl=use_translation,
            dtype=torch.float32,
            batch_size=batch_size,
            gender=gender,
            age=age,
            num_betas=num_betas,
        )

    def forward(self,
        shape:          torch.Tensor=None,
        pose:           torch.Tensor=None,
        rotation:       torch.Tensor=None,
        translation:    torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        body_output = super().forward(
            betas=shape,                    # betas -> [1, 10] # v_shaped -> [1, 10475, 3]
            body_pose=pose,                 # body_pose -> [1, 63] # joints -> [1, 118, 3]
            global_orient=rotation,         # global_orient -> [1, 3]
            transl=translation,             # transl -> [1, 3]
            pose2rot=True,
            return_full_pose=True,          # full_pose -> [1, 165] => 54 joints * 3 + 3 global rotation
            return_verts=True,              # vertices -> [1, 6890, 3]
        )
        b = body_output['vertices'].shape[0]
        return toolz.valfilter(lambda v: v is not None, {
            'vertices':     body_output['vertices'],
            'pose':         body_output['body_pose'],
            'rotation':     body_output['global_orient'],
            'translation':  body_output['transl'],
            'betas':        body_output['betas'],
            'shape':        body_output['v_shaped'],
            'joints':       body_output['joints'],
            'faces':        self.faces_tensor.expand(b, -1, -1),          #TODO: expand?
        })


class SMPLH(torch.nn.Module):
    def __init__(self,
        model_path:             str,
        joints_format:          str='none',                
        gender:                 str='neutral',
        age:                    str='adult',
        num_betas:              int=10,
        use_global_orientation: bool=True, # root rotation
        use_translation:        bool=True, # body translation
        use_pose:               bool=True, # joint rotations, False when using VPoser
        use_betas:              bool=True, # shape
        batch_size:             int=1,
    ):
        # mapper = __JOINT__MAPPERS__.get(joints_format, None)
        # super().__init__(
        #     model_path=model_path,
        #     joint_mapper=None if joints_format is None else\
        #         mapper() if mapper is not None else None,
        #     create_global_orient=use_global_orientation,
        #     create_body_pose=use_pose,
        #     create_betas=use_betas,
        #     create_transl=use_translation,
        #     dtype=torch.float32,
        #     batch_size=batch_size,
        #     gender=gender,
        #     age=age,
        #     num_betas=num_betas,
        # )
        super().__init__()
        cur_vertex_ids = vertex_ids['smplh']
        if '.npz' in model_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(model_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            data_struct.hands_componentsl = np.zeros((0))
            data_struct.hands_componentsr = np.zeros((0))
            data_struct.hands_meanl = np.zeros((15 * 3))
            data_struct.hands_meanr = np.zeros((15 * 3))
            V, D, B = data_struct.shapedirs.shape
            # data_struct.shapedirs = np.concatenate([data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM-B))], axis=-1) # super hacky way to let smplh use 16-size beta
        
        kwargs = {
                'model_type' : 'smplh',
                'data_struct' : data_struct,
                'num_betas': 10,
                'batch_size' : 64,
                'num_expression_coeffs' : 10,
                'use_pca' : False,
                'flat_hand_mean' : True,
                'vertex_ids' : cur_vertex_ids,
        }
        self.bm = smplx.SMPLH(model_path,**kwargs)

    def forward(self,
        shape:          torch.Tensor=None,
        pose:           torch.Tensor=None,
        rotation:       torch.Tensor=None,
        translation:    torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        body_output = self.bm(
            betas=shape.expand(pose.shape[0],shape.shape[1]),                    # betas -> [1, 10] # v_shaped -> [1, 10475, 3]
            body_pose=pose,                 # body_pose -> [1, 63] # joints -> [1, 118, 3]
            global_orient=rotation,         # global_orient -> [1, 3]
            transl=translation,             # transl -> [1, 3]
            pose2rot=True,
            return_full_pose=True,          # full_pose -> [1, 165] => 54 joints * 3 + 3 global rotation
            return_verts=True,              # vertices -> [1, 6890, 3]
        )
        b = body_output['vertices'].shape[0]
        return toolz.valfilter(lambda v: v is not None, {
            'vertices':     body_output['vertices'],
            'pose':         body_output['body_pose'],
            'rotation':     body_output['global_orient'],
            'translation':  body_output['transl'],
            'betas':        body_output['betas'],
            'shape':        body_output['v_shaped'],
            'joints':       body_output['joints'],
            # 'faces':        self.faces_tensor.expand(b, -1, -1),          #TODO: expand?
        })