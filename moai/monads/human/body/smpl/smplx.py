from moai.monads.human.pose.openpose import JointMap

import toolz
import torch
import smplx #TODO: try/except and error msg
import functools
import typing
import logging

#NOTE: code from https://github.com/vchoutas/smplify-x

__all__ = ["SMPLX"]

__JOINT__MAPPERS__ = {
    'none':             None,
    'openpose_coco25':  functools.partial(JointMap, 
        model='smplx', format='coco25', with_hands=True,
        with_face=True, with_face_contour=False,
    ),
    'openpose_coco25_face':  functools.partial(JointMap, 
        model='smplx', format='coco25', with_hands=True,
        with_face=True, with_face_contour=True,
    ),
    'openpose_coco19':  functools.partial(JointMap,
        model='smplx', format='coco19', with_hands=True,
        with_face=True, with_face_contour=False,
    ),
}

log = logging.getLogger(__name__)

class SMPLX(smplx.SMPLX):
    def __init__(self,
        model_path:             str,
        joints_format:          str='none',                
        gender:                 str='neutral',
        age:                    str='adult',
        num_betas:              int=10,
        pca_components:         int=6,
        use_global_orientation: bool=True, # root rotation
        use_translation:        bool=True, # body translation
        use_pose:               bool=True, # joint rotations, False when using VPoser
        use_betas:              bool=True, # shape        
        use_hands:              bool=True,
        use_face:               bool=True,
        use_eyes:               bool=True,
        use_hands_pca:          bool=True,
        flat_hand_mean:         bool=False,
        use_face_contour:       bool=False,
        batch_size:             int=1,
        # create_left_hand_pose:  bool=True,
        # create_right_hand_pose: bool=True,
        # create_expression:      bool=True,
        # create_jaw_pose:        bool=True,
        # create_left_eye_pose:   bool=True,
        # create_right_eye_pose:  bool=True,        
    ):
        mapper = __JOINT__MAPPERS__.get(joints_format, None)
        super(SMPLX, self).__init__(
            model_path=model_path,
            joint_mapper=None if joints_format is None else\
                mapper() if mapper is not None else None,
            create_global_orient=use_global_orientation,
            create_body_pose=use_pose,
            create_betas=use_betas,
            create_left_hand_pose=use_hands, # create_left_hand_pose,
            create_right_hand_pose=use_hands, # create_right_hand_pose,
            create_expression=use_face, # create_expression,
            create_jaw_pose=use_face, # create_jaw_pose,
            create_leye_pose=use_eyes, # create_left_eye_pose,
            create_reye_pose=use_eyes, # create_right_eye_pose,
            create_transl=use_translation,
            use_pca=use_hands_pca,
            dtype=torch.float32,
            batch_size=batch_size,
            gender=gender,
            age=age,
            num_pca_comps=pca_components,
            num_betas=num_betas,
            flat_hand_mean=flat_hand_mean,
            use_face_contour=use_face_contour,
        )
        log.info(f"Created a {gender} SMPL-X body model monad.")

    def forward(self,
        shape:          torch.Tensor=None,
        pose:           torch.Tensor=None,
        rotation:       torch.Tensor=None,
        translation:    torch.Tensor=None,
        left_hand:      torch.Tensor=None,
        right_hand:     torch.Tensor=None,
        expression:     torch.Tensor=None,
        jaw:            torch.Tensor=None,
        left_eye:       torch.Tensor=None,
        right_eye:      torch.Tensor=None,
    ) -> typing.Mapping[str, torch.Tensor]:
        body_output = super(SMPLX, self).forward(
            betas=shape,                    # betas -> [1, 10] # v_shaped -> [1, 10475, 3]
            body_pose=pose,                 # body_pose -> [1, 63] # joints -> [1, 118, 3]
            global_orient=rotation,         # global_orient -> [1, 3]
            transl=translation,             # transl -> [1, 3]
            left_hand_pose=left_hand,       # left_hand_pose -> [1, 45]
            right_hand_pose=right_hand,     # right_hand_pose -> [1, 45]
            expression=expression,          # expression -> [1, 10]
            jaw_pose=jaw,                   # jaw_pose -> [1, 3]
            leye_pose=left_eye,
            reye_pose=right_eye,
            pose2rot=True,
            return_full_pose=True,          # full_pose -> [1, 165] => 54 joints * 3 + 3 global rotation
            return_verts=True,              # vertices -> [1, 10475, 3]
            return_shaped=True,
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
            'left_hand':    body_output['left_hand_pose'],
            'right_hand':   body_output['right_hand_pose'],
            'expression':   body_output['expression'],
            'jaw':          body_output['jaw_pose'],
            'expressive':   body_output['full_pose'],
            'faces':        self.faces_tensor.expand(b, -1, -1),          #TODO: expand?
        })
        
        #       118 params (angle-axis) correspond to (in order)
        #           3 (global rot)
        #           + 21 (joints w/o hands) * 3
        #           + 3 (jaw)
        #           + 2 * 3 (eyes)
        #           + 2 * 15 * 3 (hands)