import functools
import typing

import smplx  # TODO: try/except and error msg
import toolz
import torch

from moai.monads.human.pose.mediapipe import JointMap

# NOTE: code from https://github.com/vchoutas/smplify-x

__all__ = ["MANO"]

__JOINT__MAPPERS__ = {
    "none": None,
    "mediapipe_hand": functools.partial(
        JointMap, model="mano", format="mediapipe_hand"
    ),
}


# NOTE: joints: https://github.com/otaheri/MANO/blob/master/mano/joints_info.py
# NOTE: see https://github.com/vchoutas/smplx/issues/48
class MANO(smplx.MANO):
    def __init__(
        self,
        model_path: str,
        joints_format: str = "none",
        is_right_hand: bool = True,
        num_betas: int = 10,
        pca_components: int = 6,
        use_global_orientation: bool = True,  # root rotation
        use_translation: bool = True,  # body translation
        use_pose: bool = True,  # joint rotations, False when using VPoser
        use_betas: bool = True,  # shape
        use_pca: bool = True,
        flat_hand_mean: bool = False,
        with_tips: bool = False,
    ):
        self.with_tips = with_tips
        super(MANO, self).__init__(
            model_path=model_path,
            joint_mapper=None,
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

    def forward(
        self,
        betas: torch.Tensor = None,
        pose: torch.Tensor = None,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
    ) -> typing.Mapping[str, torch.Tensor]:
        if pose is not None and len(pose.shape) > 2:
            pose = torch.flatten(pose, start_dim=1)
        hand_output = super(MANO, self).forward(
            betas=betas,  # betas -> [1, 10] # v_shaped -> [1, 778, 3]
            hand_pose=pose,  # body_pose -> [1, 45] or [1, 12] # joints -> [1, 15, 3]
            global_orient=rotation,  # global_orient -> [1, 3]
            transl=translation,  # transl -> [1, 3]
            return_full_pose=True,  # full_pose -> [1, 48] => 15 joints * 3 + 3 * global rotation
            return_verts=True,  # vertices -> [1, 778, 3]
        )
        b = betas.shape[0]
        if self.with_tips:
            joints = torch.cat(
                [
                    hand_output["joints"],
                    hand_output["vertices"][:, 744:745, :],  # thumb 16
                    hand_output["vertices"][:, 320:321, :],  # index 17
                    hand_output["vertices"][:, 443:444, :],  # middle 18
                    hand_output["vertices"][:, 554:555, :],  # ring 19
                    hand_output["vertices"][:, 671:672, :],  # pinky 20
                ],
                dim=-2,
            )
        else:
            joints = hand_output["joints"]
        return toolz.valfilter(
            lambda v: v is not None,
            {
                "vertices": hand_output["vertices"],
                "pose": hand_output["hand_pose"],
                "rotation": hand_output["global_orient"],
                "translation": hand_output["transl"],
                "betas": hand_output["betas"],
                "shape": hand_output["v_shaped"],
                "joints": joints,
                "full_pose": hand_output["full_pose"],
                "faces": self.faces_tensor.expand(b, -1, -1),  # faces [1, 1538 3]
            },
        )


RightHandMANO = functools.partial(MANO, is_right_hand=True)
LeftHandMANO = functools.partial(MANO, is_right_hand=False)
