from moai.monads.human.pose.openpose import JointMap

import toolz
import torch
import smplx #TODO: try/except and error msg
import functools
import typing

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
            return_verts=True,              # vertices -> [1, 10475, 3]
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