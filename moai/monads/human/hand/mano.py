from moai.monads.human.pose.openpose import JointMap

import toolz
import torch
import smplx #TODO: try/except and error msg
import functools
import typing

#NOTE: code from https://github.com/vchoutas/smplify-x

__all__ = ["MANO"]

__JOINT__MAPPERS__ = {
    'none':             None,
    'openpose_coco25':  functools.partial(JointMap, #NOTE: not implemented yet 
        model='mano', format='coco25'
    ),
    'openpose_coco19':  functools.partial(JointMap, #NOTE: not implemented yet
        model='mano', format='coco19'
    ),
}

class MANO(smplx.MANO):
    def __init__(self,
        model_path:             str,
        joints_format:          str='none',                
        is_right_hand:          bool=True,
        num_betas:              int=10,
        pca_components:         int=6,
        use_global_orientation: bool=True, # root rotation
        use_translation:        bool=True, # body translation
        use_pose:               bool=True, # joint rotations, False when using VPoser
        use_betas:              bool=True, # shape        
        use_pca:                bool=True,
        flat_hand_mean:         bool=False,       
    ):
        mapper = __JOINT__MAPPERS__.get(joints_format, None)
        super(MANO, self).__init__(
            model_path=model_path,
            joint_mapper=None if joints_format is None else\
                mapper() if mapper is not None else None,
            create_global_orient=use_global_orientation,
            create_hand_pose=use_pose,
            create_betas=use_betas,
            create_transl=use_translation,
            use_pca=use_pca,
            dtype=torch.float32,
            batch_size=1,
            num_pca_comps=pca_components,
            num_betas=num_betas,
            flat_hand_mean=flat_hand_mean,
            is_rhand=is_right_hand,
        )

    def forward(self,
        betas:          torch.Tensor=None,
        pose:           torch.Tensor=None,
        rotation:       torch.Tensor=None,
        translation:    torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        hand_output = super(MANO, self).forward(
            betas=betas,                    # betas -> [1, 10] # v_shaped -> [1, 10475, 3]
            body_pose=pose,                 # body_pose -> [1, 45] or [1, 12] # joints -> [1, 118, 3]
            global_orient=rotation,         # global_orient -> [1, 3]
            transl=translation,             # transl -> [1, 3]
            return_full_pose=True,          # full_pose -> [1, 48] => 15 joints * 3 + 3 * global rotation
            return_verts=True,              # vertices -> [1, 700, 3]
        )
        b = betas.shape[0]
        return toolz.valfilter(lambda v: v is not None, {
            'vertices':     hand_output['vertices'],
            'pose':         hand_output['hand_pose'],
            'rotation':     hand_output['global_orient'],
            'translation':  hand_output['transl'],
            'betas':        hand_output['betas'],
            'shape':        hand_output['v_shaped'],
            'joints':       hand_output['joints'],
            'full_pose':    hand_output['full_pose'],
            'faces':        self.faces_tensor.expand(b, -1, -1), # faces [1, ???, 3]
        })

RightHandMANO = functools.partial(MANO, is_right_hand=True)
LeftHandMANO = functools.partial(MANO, is_right_hand=False)